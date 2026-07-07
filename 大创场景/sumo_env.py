"""
Gym-style SUMO/TraCI environment for SAC-based VSL control.

The environment exposes a 72-dim observation built from 10-second lane-level
traffic statistics on E1-E6 and a single continuous VSL action applied to CAVs
on E1-E3.
"""

import os
import sys
import uuid
from collections import deque

import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    _sumo_home = "D:/sumo-win64-1.26.0/sumo-1.26.0"
    sys.path.append(os.path.join(_sumo_home, "tools"))

import traci
import traci.constants as tc


class RunningMeanStd:
    """Online running mean and standard deviation for state normalization."""

    def __init__(self, shape, epsilon=1e-6):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-6)


class SumoVSLEnv:
    """SUMO VSL environment with one continuous CAV speed-limit action."""

    def __init__(self, sumocfg_path, config, gui=False, seed=None, label=None):
        self.config = config
        self.sumocfg_path = sumocfg_path
        self.gui = gui
        self.seed = seed
        self.label = label or f"vsl_{uuid.uuid4().hex}"

        self.state_normalizer = RunningMeanStd(shape=(config.STATE_DIM,))
        self._prev_vsl = float(config.VSL_MAX)

        self.episode_step = 0
        self.total_steps = 0
        self.sim_time = 0.0
        self._connected = False

        self._lane_windows = {}
        self._veh_type_cache = {}
        self._reward_speed_samples = []
        self._reward_ttc_values = []

    # ================================================================
    # Public API
    # ================================================================

    def reset(self):
        """Reset SUMO and return the initial normalized observation."""
        self.close()

        detector_path = os.path.join(self.config.PROJECT_DIR, "detectors.add.xml")
        sumo_cfg = os.path.join(self.config.PROJECT_DIR, self.sumocfg_path)
        sumo_binary = "sumo-gui" if self.gui else self.config.SUMO_BINARY

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

        traci.start(sumo_cmd, label=self.label)
        traci.switch(self.label)
        self._connected = True
        self._setup_subscriptions()

        self.episode_step = 0
        self.total_steps = 0
        self.sim_time = self.config.SIM_BEGIN
        self._prev_vsl = float(self.config.VSL_MAX)
        self._veh_type_cache.clear()
        self._reset_lane_windows()
        self._reset_reward_accumulators()

        for _ in range(self.config.WARMUP_STEPS):
            self.advance_one_second(mode="none")

        self.episode_step = 0
        return self._collect_observation()

    def step(self, action):
        """Apply one VSL decision for one control interval."""
        self.set_action(action)
        self._reset_reward_accumulators()

        for _ in range(self.config.STEPS_PER_CONTROL):
            self.advance_one_second(mode="rl", collect_reward=True)
            if self.sim_time >= self.config.SIM_END:
                break

        self.episode_step += 1
        obs = self._collect_observation()
        reward = float(np.clip(self._compute_reward(), -10.0, 10.0))
        done = self.sim_time >= self.config.SIM_END
        info = {
            "sim_time": self.sim_time,
            "episode_step": self.episode_step,
            "vsl": self._prev_vsl,
            "vsl_kmh": self._prev_vsl * 3.6,
        }
        return obs, reward, done, info

    def set_action(self, action):
        """Map [-1, 1] action to VSL and enforce 20 km/h interval smoothing."""
        action_arr = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        raw_vsl = self._map_action_to_vsl(float(action_arr[0]))
        delta = np.clip(
            raw_vsl - self._prev_vsl,
            -self.config.VSL_MAX_DELTA,
            self.config.VSL_MAX_DELTA,
        )
        self._prev_vsl = float(np.clip(self._prev_vsl + delta, self.config.VSL_MIN, self.config.VSL_MAX))
        return self._prev_vsl

    def advance_one_second(self, mode="rl", collect_reward=False):
        """
        Advance one SUMO step.

        mode:
          - "rl": apply the current VSL to CAVs on E1-E3
          - "baseline": apply fixed 80 km/h to all vehicles on E1-E6
          - "none": no speed override
        """
        if mode == "rl":
            self._apply_vsl_to_cavs()
        elif mode == "baseline":
            self._apply_baseline_speed_limit()

        traci.simulationStep()
        self.sim_time = traci.simulation.getTime()
        self.total_steps += 1
        self._record_lane_sample()

        if collect_reward and self._is_sample_time(self.config.REWARD_SAMPLE_INTERVAL):
            self._sample_reward_metrics()

    def close(self):
        try:
            if self._connected:
                traci.switch(self.label)
                traci.close()
        except Exception:
            try:
                traci.close()
            except Exception:
                pass
        finally:
            self._connected = False

    # ================================================================
    # SUMO setup and state
    # ================================================================

    def _setup_subscriptions(self):
        lane_vars = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_VEHICLE_NUMBER]
        for edge in self.config.OBS_EDGES:
            for lane_idx in self.config.LANE_INDICES:
                traci.lane.subscribe(f"{edge}_{lane_idx}", lane_vars)

    def _reset_lane_windows(self):
        window_size = int(self.config.STATE_WINDOW / self.config.SUMO_STEP_LENGTH)
        self._lane_windows = {
            f"{edge}_{lane_idx}": deque(maxlen=window_size)
            for edge in self.config.OBS_EDGES
            for lane_idx in self.config.LANE_INDICES
        }

    def _record_lane_sample(self):
        lane_results = traci.lane.getAllSubscriptionResults()
        for edge in self.config.OBS_EDGES:
            edge_length = max(self.config.EDGE_LENGTHS[edge], 1e-3)
            for lane_idx in self.config.LANE_INDICES:
                lane_id = f"{edge}_{lane_idx}"
                data = lane_results.get(lane_id, {})
                speed = data.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
                veh_count = data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
                speed = max(float(speed), 0.0)
                count_per_meter = float(veh_count) / edge_length
                self._lane_windows[lane_id].append((speed, count_per_meter))

    def _collect_observation(self):
        features = []
        for edge in self.config.OBS_EDGES:
            speed_norm = max(self.config.VSL_MAX, self.config.EDGE_SPEED_LIMITS[edge], 1e-3)
            length_norm = max(self.config.EDGE_LENGTHS[edge], 1e-3)
            for lane_idx in self.config.LANE_INDICES:
                lane_id = f"{edge}_{lane_idx}"
                samples = np.asarray(self._lane_windows[lane_id], dtype=np.float32)
                if samples.size == 0:
                    speed_mean = speed_std = count_mean = count_std = 0.0
                else:
                    speed_mean = float(np.mean(samples[:, 0]))
                    speed_std = float(np.std(samples[:, 0]))
                    count_mean = float(np.mean(samples[:, 1]))
                    count_std = float(np.std(samples[:, 1]))
                features.extend([
                    speed_mean / speed_norm,
                    count_mean * length_norm / max(self.config.LANE_MAX_VEHICLES[edge], 1e-3),
                    speed_std / speed_norm,
                    count_std * length_norm / max(self.config.LANE_MAX_VEHICLES[edge], 1e-3),
                ])

        raw_state = np.asarray(features, dtype=np.float32)
        normalized_state = self.state_normalizer.normalize(raw_state)
        self.state_normalizer.update(raw_state[np.newaxis, :])
        return np.clip(normalized_state, -5.0, 5.0).astype(np.float32)

    # ================================================================
    # Action and speed control
    # ================================================================

    def _map_action_to_vsl(self, action_value):
        return self.config.VSL_MIN + 0.5 * (action_value + 1.0) * (
            self.config.VSL_MAX - self.config.VSL_MIN
        )

    def _apply_vsl_to_cavs(self):
        vsl = float(self._prev_vsl)
        for edge in self.config.CONTROL_EDGES:
            for veh_id in self._edge_vehicle_ids(edge):
                if self._vehicle_type(veh_id) in self.config.CAV_TYPES:
                    self._set_vehicle_speed(veh_id, vsl, self.config.EDGE_SPEED_LIMITS[edge])
        for edge in self.config.EVAL_EDGES:
            if edge in self.config.CONTROL_EDGES:
                continue
            for veh_id in self._edge_vehicle_ids(edge):
                if self._vehicle_type(veh_id) in self.config.CAV_TYPES:
                    self._set_vehicle_speed(veh_id, self.config.VSL_MAX, self.config.EDGE_SPEED_LIMITS[edge])

    def _apply_baseline_speed_limit(self):
        for edge in self.config.BASELINE_EDGES:
            for veh_id in self._edge_vehicle_ids(edge):
                self._set_vehicle_speed(veh_id, self.config.BASELINE_SPEED, self.config.EDGE_SPEED_LIMITS[edge])

    def _edge_vehicle_ids(self, edge):
        try:
            return traci.edge.getLastStepVehicleIDs(edge)
        except Exception:
            return []

    def _vehicle_type(self, veh_id):
        if veh_id not in self._veh_type_cache:
            try:
                self._veh_type_cache[veh_id] = traci.vehicle.getTypeID(veh_id)
            except Exception:
                self._veh_type_cache[veh_id] = "unknown"
        return self._veh_type_cache[veh_id]

    @staticmethod
    def _set_vehicle_speed(veh_id, speed, lane_speed=None):
        try:
            if lane_speed is not None and lane_speed > 0:
                traci.vehicle.setSpeedFactor(veh_id, max(float(speed) / float(lane_speed), 0.1))
            traci.vehicle.setMaxSpeed(veh_id, float(speed))
        except Exception:
            pass

    # ================================================================
    # Reward
    # ================================================================

    def _reset_reward_accumulators(self):
        self._reward_speed_samples = []
        self._reward_ttc_values = []

    def _is_sample_time(self, interval):
        return interval <= 1 or int(round(self.sim_time)) % int(interval) == 0

    def _sample_reward_metrics(self):
        speeds = []
        for edge in self.config.REWARD_EDGES:
            try:
                speed = traci.edge.getLastStepMeanSpeed(edge)
                if speed >= 0:
                    speeds.append(float(speed))
            except Exception:
                pass
            for veh_id in self._edge_vehicle_ids(edge):
                ttc = self.compute_ttc(veh_id)
                if ttc is not None and ttc < self.config.TTC_THRESHOLD:
                    self._reward_ttc_values.append(float(ttc))
        if speeds:
            self._reward_speed_samples.append(float(np.mean(speeds)))

    def _compute_reward(self):
        if self._reward_speed_samples:
            speed_kmh = float(np.mean(self._reward_speed_samples)) * 3.6
            speed_norm = speed_kmh / 120.0
        else:
            speed_norm = 0.0

        if self._reward_ttc_values:
            ttc_risk = float(np.mean(self._reward_ttc_values)) / self.config.TTC_THRESHOLD
        else:
            ttc_risk = 0.0

        return speed_norm - ttc_risk

    def compute_ttc(self, veh_id):
        """Compute Time-To-Collision for a vehicle; return None if no risk."""
        try:
            leader = traci.vehicle.getLeader(veh_id, self.config.TTC_LOOKAHEAD)
        except Exception:
            return None
        if leader is None:
            return None

        leader_id, gap = leader
        if gap <= 0:
            return 0.0

        try:
            ego_speed = traci.vehicle.getSpeed(veh_id)
            lead_speed = traci.vehicle.getSpeed(leader_id)
        except Exception:
            return None

        speed_diff = ego_speed - lead_speed
        if speed_diff <= 0:
            return None
        return gap / speed_diff
