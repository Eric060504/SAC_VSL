"""
sumo_env.py — Gym-style SUMO/TraCI environment for VSL control.

Wraps SUMO simulation with:
  - 44-dim state observation
  - 3-dim continuous action (VSL for E1, E2, E3)
  - Multi-objective reward (safety, efficiency, comfort, throughput)
  - TraCI subscriptions for efficient data retrieval
"""

import os
import sys
import math
import numpy as np
from collections import defaultdict

# Add SUMO tools to path
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    # Try default location
    _sumo_home = "D:/sumo-win64-1.26.0/sumo-1.26.0"
    sys.path.append(os.path.join(_sumo_home, "tools"))

import traci
import traci.constants as tc


class RunningMeanStd:
    """Online running mean and standard deviation using Welford's algorithm."""

    def __init__(self, shape, epsilon=1e-6):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon  # small value to avoid division by zero

    def update(self, x):
        """Update running statistics with a batch or single sample."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        # Update variance using Welford
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        """Normalize input using running statistics."""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-6)


class SumoVSLEnv:
    """
    SUMO VSL Environment using TraCI.

    State (44-dim):
      - E0-E5 edge features (6 edges × 5 features = 30)
      - E1-E3 lane features  (3 edges × 3 lanes × 1 feature = 9)
      - E0 upstream features (3)
      - Temporal encoding (2)

    Action (3-dim):
      - VSL for E1, E2, E3 ∈ [-1, 1] → mapped to [8.33, 22.22] m/s

    Reward: weighted sum of safety, efficiency, comfort, throughput
    """

    def __init__(self, sumocfg_path, config, gui=False, seed=None):
        self.config = config
        self.sumocfg_path = sumocfg_path
        self.gui = gui
        self.seed = seed

        # State normalizer
        self.state_normalizer = RunningMeanStd(shape=(config.STATE_DIM,))

        # Smoothing state
        self._prev_vsl = np.array([config.VSL_MAX] * config.ACTION_DIM, dtype=np.float32)

        # Episode tracking
        self.episode_step = 0
        self.total_steps = 0
        self.sim_time = 0.0

        # Accumulators for reward computation within one control interval
        self._interval_ttc_critical = 0
        self._interval_ttc_total = 0
        self._interval_jerk_sum = 0.0
        self._interval_jerk_count = 0
        self._interval_prev_accel = {}  # veh_id → accel at previous sim step
        self._interval_prev_arrived = 0

        # Vehicle type cache: veh_id → type_id (populated on first encounter)
        self._veh_type_cache = {}

    # ================================================================
    # Public API
    # ================================================================

    def reset(self):
        """Reset the simulation and return initial observation."""
        # Close existing connection if any
        try:
            traci.close()
        except Exception:
            pass

        # Build SUMO command
        detector_path = os.path.join(self.config.PROJECT_DIR, "detectors.add.xml")
        sumo_cfg = os.path.join(self.config.PROJECT_DIR, self.sumocfg_path)

        if self.gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = self.config.SUMO_BINARY

        sumo_cmd = [
            sumo_binary,
            "-c", sumo_cfg,
            "--additional-files", detector_path,
            "--begin", str(self.config.SIM_BEGIN),
            "--end", str(self.config.SIM_END),
            "--step-length", str(self.config.SUMO_STEP_LENGTH),
        ]
        sumo_cmd.extend(self.config.SUMO_ADDITIONAL_OPTS)

        if self.seed is not None:
            sumo_cmd.extend(["--seed", str(self.seed)])

        # Launch SUMO
        traci.start(sumo_cmd)

        # Set up subscriptions for efficient data retrieval
        self._setup_subscriptions()

        # Reset internal state
        self.episode_step = 0
        self.sim_time = self.config.SIM_BEGIN
        self._prev_vsl = np.array([self.config.VSL_MAX] * self.config.ACTION_DIM, dtype=np.float32)
        self._veh_type_cache.clear()
        self._reset_interval_accumulators()
        self._interval_prev_arrived = 0

        # Run warmup (no control)
        for _ in range(self.config.WARMUP_CONTROL_STEPS):
            self._advance_simulation_no_control()

        # Collect initial observation after warmup
        self.episode_step = self.config.WARMUP_CONTROL_STEPS
        obs = self._collect_observation()
        return obs

    def step(self, action):
        """
        Execute one control step.

        Args:
            action: np.array of shape (3,) with values in [-1, 1]
                    mapping to VSL for E1, E2, E3

        Returns:
            obs:      np.array (44,) — next state
            reward:   float — scalar reward
            done:     bool  — episode terminated
            info:     dict  — diagnostic information
        """
        # Clip action to [-1, 1]
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # EMA smoothing
        smoothed_action = (
            self.config.VSL_SMOOTHING_ALPHA * action +
            (1.0 - self.config.VSL_SMOOTHING_ALPHA) * (2.0 * (self._prev_vsl - self.config.VSL_MIN) /
                                                         (self.config.VSL_MAX - self.config.VSL_MIN) - 1.0)
        )
        self._prev_vsl = self._map_action_to_vsl(smoothed_action)

        # Reset interval accumulators
        self._reset_interval_accumulators()
        self._interval_prev_arrived = traci.simulation.getArrivedNumber()

        # Advance simulation by STEPS_PER_CONTROL steps without per-step TTC
        for _ in range(self.config.STEPS_PER_CONTROL - 1):
            self._apply_vsl_to_cavs()
            traci.simulationStep()
            self._collect_jerk_metrics()  # only jerk (cheap), no TTC

        # Last step: apply VSL, step, then sample TTC once
        self._apply_vsl_to_cavs()
        traci.simulationStep()
        self._collect_jerk_metrics()
        self._sample_ttc_once()

        self.episode_step += 1
        self.sim_time = traci.simulation.getTime()

        # Compute reward
        reward = self._compute_reward()
        reward = float(np.clip(reward, -10.0, 10.0))

        # Collect next observation
        obs = self._collect_observation()

        # Check termination
        done = self.sim_time >= self.config.SIM_END

        # Build info dict
        info = {
            "sim_time": self.sim_time,
            "episode_step": self.episode_step,
            "vsl_E1": float(self._prev_vsl[0]),
            "vsl_E2": float(self._prev_vsl[1]),
            "vsl_E3": float(self._prev_vsl[2]),
        }

        return obs, reward, done, info

    def close(self):
        """Close TraCI connection."""
        try:
            traci.close()
        except Exception:
            pass

    # ================================================================
    # Internal: SUMO setup
    # ================================================================

    def _setup_subscriptions(self):
        """Subscribe to edge and lane data for efficient batch retrieval."""
        # Edge-level subscriptions for observation edges
        edge_vars = [
            tc.LAST_STEP_VEHICLE_NUMBER,
            tc.LAST_STEP_MEAN_SPEED,
            tc.LAST_STEP_OCCUPANCY,
            tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        ]
        for edge in self.config.OBS_EDGES:
            traci.edge.subscribe(edge, edge_vars)

        # Lane-level subscriptions for control edges (per-lane speed)
        for edge in self.config.CONTROL_EDGES:
            for lane_idx in self.config.LANE_INDICES:
                lane_id = f"{edge}_{lane_idx}"
                traci.lane.subscribe(lane_id, [tc.LAST_STEP_MEAN_SPEED])

        # E3 multi-entry-exit detector subscription (travel time)
        traci.multientryexit.subscribe(self.config.E3_DETECTOR_ID, [
            tc.LAST_STEP_VEHICLE_NUMBER,
            tc.VAR_LAST_INTERVAL_TRAVELTIME,
        ])

    # ================================================================
    # Internal: Observation
    # ================================================================

    def _collect_observation(self):
        """Collect and normalize the 44-dim state observation."""
        edge_results = traci.edge.getAllSubscriptionResults()
        lane_results = traci.lane.getAllSubscriptionResults()

        features = []

        # ---- Edge-level features (E0-E5): 6 edges × 5 features = 30 dims ----
        for edge in self.config.OBS_EDGES:
            data = edge_results.get(edge, {})
            speed_limit = self.config.EDGE_SPEED_LIMITS[edge]
            edge_length = self.config.EDGE_LENGTHS[edge]
            max_veh = edge_length * self.config.LANES_PER_EDGE / self.config.VEH_EFFECTIVE_LENGTH

            mean_speed  = data.get(tc.LAST_STEP_MEAN_SPEED, -1.0)
            occupancy   = data.get(tc.LAST_STEP_OCCUPANCY, 0.0)
            veh_count   = data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            halting     = data.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)

            # F1: normalized speed ratio
            features.append(mean_speed / max(speed_limit, 1e-3))
            # F2: occupancy ratio
            features.append(occupancy / 100.0)
            # F3: density proxy
            features.append(min(veh_count / max(max_veh, 1e-3), 1.0))
            # F4: halting ratio
            features.append(halting / max(veh_count, 1))
            # F5: lane speed standard deviation
            lane_speeds = []
            for li in self.config.LANE_INDICES:
                ls = lane_results.get(f"{edge}_{li}", {})
                spd = ls.get(tc.LAST_STEP_MEAN_SPEED, -1.0)
                lane_speeds.append(spd if spd >= 0 else 0.0)
            speed_std = np.std(lane_speeds)
            features.append(speed_std / max(speed_limit, 1e-3))

        # ---- Lane-level features (E1-E3): 3 edges × 3 lanes × 1 = 9 dims ----
        for edge in self.config.CONTROL_EDGES:
            for li in self.config.LANE_INDICES:
                ls = lane_results.get(f"{edge}_{li}", {})
                spd = ls.get(tc.LAST_STEP_MEAN_SPEED, -1.0)
                features.append(max(spd, 0.0) / 22.22)

        # ---- Upstream E0 features: 3 dims ----
        e0_data = edge_results.get(self.config.UPSTREAM_EDGE, {})
        e0_count = e0_data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
        e0_max = (self.config.EDGE_LENGTHS["E0"] * self.config.LANES_PER_EDGE /
                  self.config.VEH_EFFECTIVE_LENGTH)
        features.append(min(e0_count / max(e0_max, 1e-3), 1.0))       # inflow ratio
        features.append(e0_data.get(tc.LAST_STEP_MEAN_SPEED, 0.0) / max(33.30, 1e-3))  # speed ratio
        features.append(e0_data.get(tc.LAST_STEP_OCCUPANCY, 0.0) / 100.0)               # occupancy

        # ---- Temporal encoding: 2 dims ----
        sim_step_normalized = self.sim_time / self.config.SIM_END  # [0, 1]
        features.append(math.sin(2.0 * math.pi * sim_step_normalized))
        features.append(math.cos(2.0 * math.pi * sim_step_normalized))

        # Convert to numpy and normalize (normalize first, then update running stats)
        raw_state = np.array(features, dtype=np.float32)
        normalized_state = self.state_normalizer.normalize(raw_state)
        # Update running statistics after normalizing current sample
        self.state_normalizer.update(raw_state[np.newaxis, :])

        return np.clip(normalized_state, -5.0, 5.0).astype(np.float32)

    # ================================================================
    # Internal: Action
    # ================================================================

    def _map_action_to_vsl(self, action):
        """Map action from [-1, 1] to VSL in [VSL_MIN, VSL_MAX] m/s."""
        return self.config.VSL_MIN + 0.5 * (action + 1.0) * (self.config.VSL_MAX - self.config.VSL_MIN)

    def _apply_vsl_to_cavs(self):
        """Apply current VSL values to CAV vehicles on control edges."""
        for i, edge in enumerate(self.config.CONTROL_EDGES):
            vsl = float(self._prev_vsl[i])
            try:
                veh_ids = traci.edge.getLastStepVehicleIDs(edge)
            except Exception:
                continue
            for veh_id in veh_ids:
                # Check vehicle type (with caching)
                if veh_id not in self._veh_type_cache:
                    try:
                        self._veh_type_cache[veh_id] = traci.vehicle.getTypeID(veh_id)
                    except Exception:
                        self._veh_type_cache[veh_id] = "unknown"
                vtype = self._veh_type_cache[veh_id]
                if vtype in self.config.CAV_TYPES:
                    try:
                        traci.vehicle.setMaxSpeed(veh_id, vsl)
                    except Exception:
                        pass

    # ================================================================
    # Internal: Simulation Advance
    # ================================================================

    def _advance_simulation_no_control(self):
        """Advance simulation by one control interval without applying VSL."""
        for _ in range(self.config.STEPS_PER_CONTROL):
            traci.simulationStep()

    def _collect_jerk_metrics(self):
        """Cheap per-step metrics: only jerk for CAVs (no getLeader calls)."""
        for edge in self.config.REWARD_EDGES:
            try:
                veh_ids = traci.edge.getLastStepVehicleIDs(edge)
            except Exception:
                continue
            for veh_id in veh_ids:
                if veh_id not in self._veh_type_cache:
                    try:
                        self._veh_type_cache[veh_id] = traci.vehicle.getTypeID(veh_id)
                    except Exception:
                        self._veh_type_cache[veh_id] = "unknown"
                if self._veh_type_cache.get(veh_id, "") in self.config.CAV_TYPES:
                    try:
                        accel = traci.vehicle.getAcceleration(veh_id)
                    except Exception:
                        accel = 0.0
                    if veh_id in self._interval_prev_accel:
                        prev_accel = self._interval_prev_accel[veh_id]
                        jerk = abs(accel - prev_accel)
                        self._interval_jerk_sum += jerk
                        self._interval_jerk_count += 1
                    self._interval_prev_accel[veh_id] = accel

    def _sample_ttc_once(self):
        """Sample TTC at control interval end, max 100 vehicles on reward edges."""
        reward_edges = self.config.REWARD_EDGES
        all_veh_ids = []
        for edge in reward_edges:
            try:
                all_veh_ids.extend(traci.edge.getLastStepVehicleIDs(edge))
            except Exception:
                pass

        # Cap at 100 vehicles for speed
        if len(all_veh_ids) > 100:
            import random
            all_veh_ids = random.sample(all_veh_ids, 100)

        self._interval_ttc_total = len(all_veh_ids)
        for veh_id in all_veh_ids:
            ttc = self._compute_ttc(veh_id)
            if ttc is not None and ttc < self.config.TTC_THRESHOLD:
                self._interval_ttc_critical += 1

    def _compute_ttc(self, veh_id):
        """
        Compute Time-To-Collision for a given vehicle.

        Returns TTC in seconds, or None if no collision risk.
        """
        try:
            leader = traci.vehicle.getLeader(veh_id, self.config.TTC_LOOKAHEAD)
        except Exception:
            return None

        if leader is None:
            return None

        leader_id, gap = leader
        if gap <= 0:
            return 0.0  # immediate collision risk

        try:
            ego_speed = traci.vehicle.getSpeed(veh_id)
            lead_speed = traci.vehicle.getSpeed(leader_id)
        except Exception:
            return None

        speed_diff = ego_speed - lead_speed
        if speed_diff <= 0:
            return None  # ego not faster than leader, no risk

        return gap / speed_diff

    def _reset_interval_accumulators(self):
        """Reset accumulators for the next control interval."""
        self._interval_ttc_critical = 0
        self._interval_ttc_total = 0
        self._interval_jerk_sum = 0.0
        self._interval_jerk_count = 0
        self._interval_prev_accel.clear()

    # ================================================================
    # Internal: Reward
    # ================================================================

    def _compute_reward(self):
        """Compute the multi-objective reward for the current interval."""
        r_safety     = self._compute_r_safety()
        r_efficiency = self._compute_r_efficiency()
        r_comfort    = self._compute_r_comfort()
        r_throughput = self._compute_r_throughput()

        total = (self.config.W_SAFETY     * r_safety +
                 self.config.W_EFFICIENCY * r_efficiency +
                 self.config.W_COMFORT    * r_comfort +
                 self.config.W_THROUGHPUT * r_throughput)

        return total

    def _compute_r_safety(self):
        """Safety reward based on TTC violations and speed variance."""
        # TTC penalty
        ttc_ratio = (self._interval_ttc_critical /
                     max(self._interval_ttc_total, 1))
        penalty_ttc = min(ttc_ratio * 5.0, 0.5)

        # Speed variance penalty across lanes
        lane_results = traci.lane.getAllSubscriptionResults()
        total_variance = 0.0
        for edge in self.config.REWARD_EDGES:
            lane_speeds = []
            for li in self.config.LANE_INDICES:
                ls = lane_results.get(f"{edge}_{li}", {})
                spd = ls.get(tc.LAST_STEP_MEAN_SPEED, -1.0)
                lane_speeds.append(spd if spd >= 0 else 0.0)
            std_dev = np.std(lane_speeds)
            total_variance += std_dev / 22.22  # normalize by speed limit

        avg_variance = total_variance / len(self.config.REWARD_EDGES)
        penalty_variance = min(avg_variance * 2.0, 0.4)

        return max(1.0 - penalty_ttc - penalty_variance, 0.0)

    def _compute_r_efficiency(self):
        """Efficiency reward based on E3 travel time detector."""
        try:
            mean_tt = traci.multientryexit.getLastIntervalMeanTravelTime(
                self.config.E3_DETECTOR_ID)
        except Exception:
            return 0.5  # neutral fallback

        if mean_tt is None or mean_tt <= 0:
            return 0.5

        free_flow = self.config.FREEFLOW_TRAVEL_TIME

        if mean_tt <= free_flow:
            return 1.0

        penalty = min((mean_tt - free_flow) / free_flow, 1.0)
        return 1.0 - penalty

    def _compute_r_comfort(self):
        """Comfort reward based on jerk (acceleration change rate)."""
        if self._interval_jerk_count == 0:
            return 1.0  # no CAV observations, assume comfortable

        avg_jerk = self._interval_jerk_sum / self._interval_jerk_count
        return math.exp(-avg_jerk / max(self.config.JERK_REFERENCE, 1e-3))

    def _compute_r_throughput(self):
        """Throughput reward based on completed vehicle count."""
        arrived = traci.simulation.getArrivedNumber()
        arrived_this_interval = arrived - self._interval_prev_arrived

        expected = self.config.TRAFFIC_FLOW_RATE * self.config.CONTROL_INTERVAL / 3600.0  # ≈ 56.67
        throughput_rate = arrived_this_interval / max(expected, 1e-3)

        return min(throughput_rate, 1.5) / 1.5
