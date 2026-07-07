"""
Training and evaluation orchestration for SAC-based VSL control.
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "D:/sumo-win64-1.26.0/sumo-1.26.0"
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

from sac_agent import ReplayBuffer, SACAgent
from sumo_env import SumoVSLEnv


class SACVSLTrainer:
    """Trainer that ties together the SUMO environment and SAC agent."""

    def __init__(self, config, scenario_name, gui=False, checkpoint_path=None):
        self.config = config
        self.scenario_name = scenario_name
        self.sumocfg, self.cav_rate = config.SCENARIOS[scenario_name]
        self.gui = gui
        self.checkpoint_path = checkpoint_path

        seed = config.SCENARIO_SEEDS.get(scenario_name, 0)
        np.random.seed(seed)

        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.env = SumoVSLEnv(self.sumocfg, config, gui=gui, seed=seed)
        self.agent = SACAgent(config.STATE_DIM, config.ACTION_DIM, config)
        self.buffer = ReplayBuffer(config.STATE_DIM, config.ACTION_DIM, max_size=config.BUFFER_SIZE)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.agent.load(checkpoint_path)

        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.alphas = []
        self.vsl_history = []
        self.eval_metrics = {}

    # ================================================================
    # Training
    # ================================================================

    def train(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.config.TRAIN_EPISODES

        total_steps = 0
        best_avg_reward = -float("inf")
        checkpoint_dir = self.config.CHECKPOINT_DIR

        print(f"\n{'=' * 60}")
        print(f"Training SAC VSL: scenario={self.scenario_name} "
              f"(CAV={self.cav_rate:.0%}), episodes={num_episodes}")
        print(f"{'=' * 60}")

        for episode in range(1, num_episodes + 1):
            episode_start_time = time.time()
            obs = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = defaultdict(list)
            info = {"vsl": self.config.VSL_MAX, "vsl_kmh": self.config.VSL_MAX * 3.6}

            while True:
                action = self.agent.select_action(obs, evaluate=False)
                next_obs, reward, done, info = self.env.step(action)

                self.buffer.push(
                    obs,
                    action,
                    np.array([reward], dtype=np.float32),
                    next_obs,
                    np.array([float(done)], dtype=np.float32),
                )

                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE:
                    for _ in range(self.config.UPDATES_PER_STEP):
                        loss_info = self.agent.update(self.buffer)
                    for key, value in loss_info.items():
                        episode_losses[key].append(value)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            avg_losses = {k: np.mean(v) for k, v in episode_losses.items()} if episode_losses else {}
            self.actor_losses.append(avg_losses.get("actor_loss", 0.0))
            self.critic_losses.append(avg_losses.get("critic_loss", 0.0))
            self.alphas.append(self.agent.alpha_value)
            self.vsl_history.append(info.get("vsl", 0.0))

            elapsed = time.time() - episode_start_time
            if episode % self.config.LOG_INTERVAL == 0:
                print(f"Ep {episode:4d}/{num_episodes} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Steps: {episode_steps:3d} | "
                      f"alpha: {self.agent.alpha_value:.4f} | "
                      f"ActorLoss: {avg_losses.get('actor_loss', 0):.4f} | "
                      f"CriticLoss: {avg_losses.get('critic_loss', 0):.4f} | "
                      f"Time: {elapsed:.1f}s | "
                      f"VSL[E1-E3]: {info.get('vsl', 0):.2f} m/s "
                      f"({info.get('vsl_kmh', 0):.1f} km/h)")

            if episode % self.config.SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"sac_vsl_{self.scenario_name}_ep{episode}.pt")
                self.agent.save(ckpt_path)
                print(f"  -> Checkpoint saved: {ckpt_path}")

            recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
            if len(self.episode_rewards) >= 10 and recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                best_path = os.path.join(checkpoint_dir, f"sac_vsl_{self.scenario_name}_best.pt")
                self.agent.save(best_path)

        final_path = os.path.join(checkpoint_dir, f"sac_vsl_{self.scenario_name}_final.pt")
        self.agent.save(final_path)
        print(f"\nTraining complete. Final model: {final_path}")
        print(f"Best avg reward (over 10 eps): {best_avg_reward:.2f}")
        self.env.close()

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self, num_episodes=None):
        """Evaluate deterministic SAC VSL and fixed 80 km/h no-control baseline."""
        if num_episodes is None:
            num_episodes = self.config.EVAL_EPISODES

        print(f"\n{'=' * 60}")
        print(f"Evaluating scenario={self.scenario_name} "
              f"(CAV={self.cav_rate:.0%}), episodes={num_episodes}")
        print(f"{'=' * 60}")

        rl_runs = []
        baseline_runs = []
        for episode in range(1, num_episodes + 1):
            rl_metrics = self._evaluate_one_episode(policy="rl")
            baseline_metrics = self._evaluate_one_episode(policy="baseline")
            rl_runs.append(rl_metrics)
            baseline_runs.append(baseline_metrics)
            print(f"  Eval Ep {episode:2d} | "
                  f"RL TT={rl_metrics['total_travel_time']:.1f}s, "
                  f"CO2={rl_metrics['co2_emission']:.1f}mg, "
                  f"TTC={rl_metrics['ttc_total']} | "
                  f"Baseline TT={baseline_metrics['total_travel_time']:.1f}s, "
                  f"CO2={baseline_metrics['co2_emission']:.1f}mg, "
                  f"TTC={baseline_metrics['ttc_total']}")

        metrics = {
            "scenario": self.scenario_name,
            "cav_rate": self.cav_rate,
            "rl": self._aggregate_eval_metrics(rl_runs),
            "baseline": self._aggregate_eval_metrics(baseline_runs),
        }
        self.eval_metrics = metrics
        self.env.close()

        print(f"\n  === Summary ({self.scenario_name}, CAV={self.cav_rate:.0%}) ===")
        self._print_eval_summary("RL", metrics["rl"])
        self._print_eval_summary("Baseline 80km/h", metrics["baseline"])
        return metrics

    def _evaluate_one_episode(self, policy):
        import traci

        obs = self.env.reset()
        reward_sum = 0.0
        reward_count = 0
        vsl_values = []
        trip_entries = {}
        completed_travel_times = []
        co2_emission = 0.0
        ttc_total = 0
        ttc_e4 = 0
        ttc_e1_e3 = 0
        e4_speeds = []
        e1_e3_speeds = []

        while self.env.sim_time < self.config.SIM_END:
            if policy == "rl":
                action = self.agent.select_action(obs, evaluate=True)
                self.env.set_action(action)
                self.env._reset_reward_accumulators()
                vsl_values.append(self.env._prev_vsl)
                mode = "rl"
            else:
                mode = "baseline"

            for _ in range(self.config.STEPS_PER_CONTROL):
                if self.env.sim_time >= self.config.SIM_END:
                    break

                self.env.advance_one_second(mode=mode, collect_reward=(policy == "rl"))
                self._collect_travel_time_sample(traci, trip_entries, completed_travel_times)
                self._collect_speed_samples(traci, e4_speeds, e1_e3_speeds)
                co2_emission += self._collect_co2(traci)
                counts = self._collect_ttc_counts(traci)
                ttc_total += counts["total"]
                ttc_e4 += counts["e4"]
                ttc_e1_e3 += counts["e1_e3"]

            if policy == "rl":
                reward_sum += self.env._compute_reward()
                reward_count += 1
                obs = self.env._collect_observation()

        self.env.close()
        return {
            "mean_reward": reward_sum / max(reward_count, 1),
            "total_travel_time": float(np.sum(completed_travel_times)),
            "co2_emission": float(co2_emission),
            "ttc_total": int(ttc_total),
            "ttc_e4": int(ttc_e4),
            "e4_mean_speed": float(np.mean(e4_speeds)) if e4_speeds else 0.0,
            "ttc_e1_e3": int(ttc_e1_e3),
            "e1_e3_mean_speed": float(np.mean(e1_e3_speeds)) if e1_e3_speeds else 0.0,
            "avg_vsl": float(np.mean(vsl_values)) if vsl_values else self.config.BASELINE_SPEED,
        }

    @staticmethod
    def _edge_vehicle_ids(traci, edge):
        try:
            return traci.edge.getLastStepVehicleIDs(edge)
        except Exception:
            return []

    def _collect_travel_time_sample(self, traci, trip_entries, completed_travel_times):
        sim_time = traci.simulation.getTime()
        for veh_id in self._edge_vehicle_ids(traci, "E1"):
            trip_entries.setdefault(veh_id, sim_time)
        for veh_id in self._edge_vehicle_ids(traci, "E7"):
            start_time = trip_entries.pop(veh_id, None)
            if start_time is not None:
                completed_travel_times.append(max(sim_time - start_time, 0.0))

    def _collect_speed_samples(self, traci, e4_speeds, e1_e3_speeds):
        try:
            e4_speed = traci.edge.getLastStepMeanSpeed(self.config.E4_EDGE)
            if e4_speed >= 0:
                e4_speeds.append(float(e4_speed))
        except Exception:
            pass

        speeds = []
        for edge in self.config.E1_E3_EDGES:
            try:
                speed = traci.edge.getLastStepMeanSpeed(edge)
                if speed >= 0:
                    speeds.append(float(speed))
            except Exception:
                pass
        if speeds:
            e1_e3_speeds.append(float(np.mean(speeds)))

    @staticmethod
    def _collect_co2(traci):
        total = 0.0
        try:
            veh_ids = traci.vehicle.getIDList()
        except Exception:
            return total
        for veh_id in veh_ids:
            try:
                total += float(traci.vehicle.getCO2Emission(veh_id))
            except Exception:
                pass
        return total

    def _collect_ttc_counts(self, traci):
        counts = {"total": 0, "e4": 0, "e1_e3": 0}
        for edge in self.config.EVAL_EDGES:
            for veh_id in self._edge_vehicle_ids(traci, edge):
                ttc = self.env.compute_ttc(veh_id)
                if ttc is not None and ttc < self.config.TTC_THRESHOLD:
                    counts["total"] += 1
                    if edge == self.config.E4_EDGE:
                        counts["e4"] += 1
                    if edge in self.config.E1_E3_EDGES:
                        counts["e1_e3"] += 1
        return counts

    @staticmethod
    def _aggregate_eval_metrics(runs):
        keys = [
            "mean_reward",
            "total_travel_time",
            "co2_emission",
            "ttc_total",
            "ttc_e4",
            "e4_mean_speed",
            "ttc_e1_e3",
            "e1_e3_mean_speed",
            "avg_vsl",
        ]
        return {key: float(np.mean([run[key] for run in runs])) for key in keys}

    @staticmethod
    def _print_eval_summary(label, metrics):
        print(f"  [{label}]")
        print(f"    Total travel time E1-E6: {metrics['total_travel_time']:.1f} s")
        print(f"    CO2 emission:            {metrics['co2_emission']:.1f} mg")
        print(f"    TTC<3s total:            {metrics['ttc_total']:.1f}")
        print(f"    TTC<3s E4:               {metrics['ttc_e4']:.1f}")
        print(f"    Mean speed E4:           {metrics['e4_mean_speed']:.2f} m/s")
        print(f"    TTC<3s E1-E3:            {metrics['ttc_e1_e3']:.1f}")
        print(f"    Mean speed E1-E3:        {metrics['e1_e3_mean_speed']:.2f} m/s")
        print(f"    Avg VSL:                 {metrics['avg_vsl']:.2f} m/s")

    # ================================================================
    # Plotting
    # ================================================================

    def plot_results(self, save_dir=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = os.path.join(self.config.PROJECT_DIR, "results")
        os.makedirs(save_dir, exist_ok=True)
        name = self.scenario_name

        episodes = np.arange(1, len(self.episode_rewards) + 1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        rewards = np.asarray(self.episode_rewards)
        ax.plot(episodes, rewards, "b-", alpha=0.3, label="Episode Reward")
        if len(rewards) >= 10:
            smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
            ax.plot(episodes[:len(smooth)], smooth, "r-", linewidth=2, label="MA10")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Training Reward - {name} (CAV={self.cav_rate:.0%})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"reward_curve_{name}.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(episodes, self.actor_losses, "b-", alpha=0.5)
        axes[0].set_ylabel("Actor Loss")
        axes[1].plot(episodes, self.critic_losses, "r-", alpha=0.5)
        axes[1].set_ylabel("Critic Loss")
        axes[2].plot(episodes, self.alphas, "g-", linewidth=2)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Alpha")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"loss_curve_{name}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        if self.vsl_history:
            vsl = np.asarray(self.vsl_history)
            ax.plot(episodes, vsl, "b-", label="Unified E1-E3 VSL", linewidth=1.5)
            ax.axhline(y=self.config.VSL_MAX, color="gray", linestyle="--", alpha=0.5, label="120 km/h")
            ax.axhline(y=self.config.VSL_MIN, color="gray", linestyle=":", alpha=0.5, label="60 km/h")
        ax.set_xlabel("Episode")
        ax.set_ylabel("VSL (m/s)")
        ax.set_title(f"VSL Policy Evolution - {name} (CAV={self.cav_rate:.0%})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"vsl_policy_{name}.png"), dpi=150)
        plt.close(fig)

        print(f"  -> Plots saved to {save_dir}")

    @staticmethod
    def plot_comparison(all_metrics, save_dir=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = os.path.join("results")
        os.makedirs(save_dir, exist_ok=True)

        labels = [m["scenario"] for m in all_metrics]
        rl_tt = [m["rl"]["total_travel_time"] for m in all_metrics]
        base_tt = [m["baseline"]["total_travel_time"] for m in all_metrics]
        rl_co2 = [m["rl"]["co2_emission"] for m in all_metrics]
        base_co2 = [m["baseline"]["co2_emission"] for m in all_metrics]
        rl_ttc = [m["rl"]["ttc_total"] for m in all_metrics]
        base_ttc = [m["baseline"]["ttc_total"] for m in all_metrics]

        x = np.arange(len(labels))
        width = 0.35
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, rl_vals, base_vals, title, ylabel in [
            (axes[0], rl_tt, base_tt, "Total Travel Time E1-E6", "s"),
            (axes[1], rl_co2, base_co2, "CO2 Emission", "mg"),
            (axes[2], rl_ttc, base_ttc, "TTC<3s Total", "count"),
        ]:
            ax.bar(x - width / 2, rl_vals, width, label="RL")
            ax.bar(x + width / 2, base_vals, width, label="Baseline")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "scenario_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Comparison plot saved to {save_dir}")


if __name__ == "__main__":
    import config as cfg

    trainer = SACVSLTrainer(cfg, "cav100", gui=False)
    trainer.evaluate(num_episodes=1)
