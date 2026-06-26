"""
sac_vsl.py — Training orchestrator for SAC-based VSL control.

Handles:
  - Training loop over multiple episodes
  - Evaluation with deterministic policy
  - Model checkpointing
  - Logging of training metrics
"""

import os
import sys
import time
import numpy as np
from collections import defaultdict

# Ensure SUMO tools are on the path
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "D:/sumo-win64-1.26.0/sumo-1.26.0"
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

from sumo_env import SumoVSLEnv
from sac_agent import SACAgent, ReplayBuffer


class SACVSLTrainer:
    """
    Trainer that ties together the SUMO VSL environment and SAC agent.
    """

    def __init__(self, config, scenario_name, gui=False, checkpoint_path=None):
        self.config = config
        self.scenario_name = scenario_name
        self.sumocfg, self.cav_rate = config.SCENARIOS[scenario_name]
        self.gui = gui
        self.checkpoint_path = checkpoint_path

        # Set seed
        seed = config.SCENARIO_SEEDS.get(scenario_name, 0)
        np.random.seed(seed)
        torch_seed = seed

        import torch
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)

        # Create environment
        self.env = SumoVSLEnv(self.sumocfg, config, gui=gui, seed=seed)

        # Create agent
        self.agent = SACAgent(config.STATE_DIM, config.ACTION_DIM, config)

        # Create replay buffer
        self.buffer = ReplayBuffer(config.STATE_DIM, config.ACTION_DIM,
                                   max_size=config.BUFFER_SIZE)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.agent.load(checkpoint_path)

        # Metrics history
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.alphas = []
        self.vsl_history = []  # list of avg VSL per episode
        self.eval_metrics = []

    # ================================================================
    # Training
    # ================================================================

    def train(self, num_episodes=None):
        """
        Run the training loop.

        Args:
            num_episodes: number of episodes (defaults to config.TRAIN_EPISODES)
        """
        if num_episodes is None:
            num_episodes = self.config.TRAIN_EPISODES

        total_steps = 0
        best_avg_reward = -float("inf")
        checkpoint_dir = self.config.CHECKPOINT_DIR

        print(f"\n{'='*60}")
        print(f"Training SAC VSL: scenario={self.scenario_name} "
              f"(CAV={self.cav_rate:.0%}), episodes={num_episodes}")
        print(f"{'='*60}")

        for episode in range(1, num_episodes + 1):
            episode_start_time = time.time()

            # Reset environment
            obs = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = defaultdict(list)

            # Run one episode
            while True:
                # Select action
                action = self.agent.select_action(obs, evaluate=False)

                # Step environment
                next_obs, reward, done, info = self.env.step(action)

                # Store transition
                self.buffer.push(obs, action, np.array([reward], dtype=np.float32),
                                 next_obs, np.array([float(done)], dtype=np.float32))

                # Update agent if buffer has enough data
                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE:
                    for _ in range(self.config.UPDATES_PER_STEP):
                        loss_info = self.agent.update(self.buffer)
                    for k, v in loss_info.items():
                        episode_losses[k].append(v)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if done:
                    break

            # End of episode
            self.episode_rewards.append(episode_reward)
            avg_losses = {k: np.mean(v) for k, v in episode_losses.items()} if episode_losses else {}
            self.actor_losses.append(avg_losses.get('actor_loss', 0.0))
            self.critic_losses.append(avg_losses.get('critic_loss', 0.0))
            self.alphas.append(self.agent.alpha_value)
            self.vsl_history.append([
                info.get('vsl_E1', 0), info.get('vsl_E2', 0), info.get('vsl_E3', 0)
            ])

            elapsed = time.time() - episode_start_time

            # Logging
            if episode % self.config.LOG_INTERVAL == 0:
                print(f"Ep {episode:4d}/{num_episodes} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Steps: {episode_steps:3d} | "
                      f"α: {self.agent.alpha_value:.4f} | "
                      f"ActorLoss: {avg_losses.get('actor_loss', 0):.4f} | "
                      f"CriticLoss: {avg_losses.get('critic_loss', 0):.4f} | "
                      f"Time: {elapsed:.1f}s | "
                      f"VSL[E1/E2/E3]: {info.get('vsl_E1',0):.1f}/{info.get('vsl_E2',0):.1f}/{info.get('vsl_E3',0):.1f}")

            # Periodic checkpoint
            if episode % self.config.SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(checkpoint_dir,
                                         f"sac_vsl_{self.scenario_name}_ep{episode}.pt")
                self.agent.save(ckpt_path)
                print(f"  -> Checkpoint saved: {ckpt_path}")

            # Track best model
            recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
            if len(self.episode_rewards) >= 10 and recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                best_path = os.path.join(checkpoint_dir, f"sac_vsl_{self.scenario_name}_best.pt")
                self.agent.save(best_path)

        # Save final model
        final_path = os.path.join(checkpoint_dir, f"sac_vsl_{self.scenario_name}_final.pt")
        self.agent.save(final_path)
        print(f"\nTraining complete. Final model: {final_path}")
        print(f"Best avg reward (over 10 eps): {best_avg_reward:.2f}")

        self.env.close()

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self, num_episodes=None):
        """
        Evaluate the trained agent with deterministic policy.

        Returns:
            dict of evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.config.EVAL_EPISODES

        print(f"\n{'='*60}")
        print(f"Evaluating SAC VSL: scenario={self.scenario_name} "
              f"(CAV={self.cav_rate:.0%}), episodes={num_episodes}")
        print(f"{'='*60}")

        all_rewards = []
        all_ttc_violations = []
        all_avg_speeds = []
        all_travel_times = []
        all_vsl_values = []

        import traci
        import traci.constants as tc

        for ep in range(1, num_episodes + 1):
            obs = self.env.reset()
            ep_reward = 0.0
            ep_ttc = 0
            ep_speeds = []
            ep_tt = []
            ep_vsls = []

            # Collect CAV-specific metrics in evaluation mode
            steps_in_ep = 0
            while True:
                action = self.agent.select_action(obs, evaluate=True)
                # Record VSL values
                vsl_raw = self.env._prev_vsl.copy()
                ep_vsls.append(vsl_raw)

                next_obs, reward, done, info = self.env.step(action)
                obs = next_obs
                ep_reward += reward
                steps_in_ep += 1

                # Collect TTC violations (same computation as in env)
                try:
                    for edge in self.config.REWARD_EDGES:
                        veh_ids = traci.edge.getLastStepVehicleIDs(edge)
                        for veh_id in veh_ids:
                            ttc = self.env._compute_ttc(veh_id)
                            if ttc is not None and ttc < self.config.TTC_THRESHOLD:
                                ep_ttc += 1
                except Exception:
                    pass

                # Collect edge-level speeds on E1-E5
                try:
                    for edge in self.config.REWARD_EDGES:
                        spd = traci.edge.getLastStepMeanSpeed(edge)
                        if spd >= 0:
                            ep_speeds.append(spd)
                except Exception:
                    pass

                # Collect E3 travel time
                try:
                    mean_tt = traci.multientryexit.getLastIntervalMeanTravelTime(
                        self.config.E3_DETECTOR_ID)
                    if mean_tt is not None and mean_tt > 0:
                        ep_tt.append(float(mean_tt))
                except Exception:
                    pass

                if done:
                    break

            all_rewards.append(ep_reward)
            all_ttc_violations.append(ep_ttc)
            if ep_speeds:
                all_avg_speeds.append(np.mean(ep_speeds))
            if ep_tt:
                all_travel_times.append(np.mean(ep_tt))
            ep_vsls = np.array(ep_vsls)
            all_vsl_values.append(ep_vsls.mean(axis=0))

            print(f"  Eval Ep {ep:2d}: reward={ep_reward:7.2f}, "
                  f"TTC<3s={ep_ttc:5d}, "
                  f"avg_speed={np.mean(ep_speeds) if ep_speeds else 0:.2f} m/s, "
                  f"avg_TT={np.mean(ep_tt) if ep_tt else 0:.1f} s")

        # Aggregate metrics
        metrics = {
            "scenario": self.scenario_name,
            "cav_rate": self.cav_rate,
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_ttc_violations": float(np.mean(all_ttc_violations)),
            "mean_speed": float(np.mean(all_avg_speeds)) if all_avg_speeds else 0.0,
            "mean_travel_time": float(np.mean(all_travel_times)) if all_travel_times else 0.0,
            "avg_vsl_E1": float(np.mean([v[0] for v in all_vsl_values])),
            "avg_vsl_E2": float(np.mean([v[1] for v in all_vsl_values])),
            "avg_vsl_E3": float(np.mean([v[2] for v in all_vsl_values])),
        }

        self.eval_metrics = metrics
        self.env.close()

        print(f"\n  === Summary ({self.scenario_name}, CAV={self.cav_rate:.0%}) ===")
        print(f"  Mean Reward:        {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean TTC Violations: {metrics['mean_ttc_violations']:.1f}")
        print(f"  Mean Speed:          {metrics['mean_speed']:.2f} m/s")
        print(f"  Mean Travel Time:    {metrics['mean_travel_time']:.1f} s")
        print(f"  Avg VSL [E1/E2/E3]:  {metrics['avg_vsl_E1']:.1f}/{metrics['avg_vsl_E2']:.1f}/{metrics['avg_vsl_E3']:.1f} m/s")

        return metrics

    # ================================================================
    # Plotting
    # ================================================================

    def plot_results(self, save_dir=None):
        """Generate and save training result plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = os.path.join(self.config.PROJECT_DIR, "results")
        os.makedirs(save_dir, exist_ok=True)
        name = self.scenario_name

        # --- Reward curve ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        rewards = np.array(self.episode_rewards)
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')
        if len(rewards) >= 10:
            kernel = np.ones(10) / 10
            smooth = np.convolve(rewards, kernel, mode='valid')
            ax.plot(episodes[:len(smooth)], smooth, 'r-', linewidth=2, label='Smoothed (MA10)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'Training Reward — {name} (CAV={self.cav_rate:.0%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'reward_curve_{name}.png'), dpi=150)
        plt.close(fig)

        # --- Loss & alpha curve ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(episodes, self.actor_losses, 'b-', alpha=0.3)
        if len(self.actor_losses) >= 5:
            smooth_al = np.convolve(self.actor_losses, np.ones(5)/5, mode='valid')
            axes[0].plot(episodes[:len(smooth_al)], smooth_al, 'b-', linewidth=2)
        axes[0].set_ylabel('Actor Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, self.critic_losses, 'r-', alpha=0.3)
        if len(self.critic_losses) >= 5:
            smooth_cl = np.convolve(self.critic_losses, np.ones(5)/5, mode='valid')
            axes[1].plot(episodes[:len(smooth_cl)], smooth_cl, 'r-', linewidth=2)
        axes[1].set_ylabel('Critic Loss')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(episodes, self.alphas, 'g-', linewidth=2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Alpha (entropy coeff)')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'Loss & Alpha — {name} (CAV={self.cav_rate:.0%})')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'loss_curve_{name}.png'), dpi=150)
        plt.close(fig)

        # --- VSL policy curve ---
        vsl = np.array(self.vsl_history)
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(vsl) > 0:
            ax.plot(episodes, vsl[:, 0], 'r-', label='E1 VSL', linewidth=1.5)
            ax.plot(episodes, vsl[:, 1], 'g-', label='E2 VSL', linewidth=1.5)
            ax.plot(episodes, vsl[:, 2], 'b-', label='E3 VSL', linewidth=1.5)
            ax.axhline(y=22.22, color='gray', linestyle='--', alpha=0.5, label='80 km/h limit')
            ax.axhline(y=8.33, color='gray', linestyle=':', alpha=0.5, label='30 km/h min')
        ax.set_xlabel('Episode')
        ax.set_ylabel('VSL (m/s)')
        ax.set_title(f'VSL Policy Evolution — {name} (CAV={self.cav_rate:.0%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'vsl_policy_{name}.png'), dpi=150)
        plt.close(fig)

        print(f"  -> Plots saved to {save_dir}")

    @staticmethod
    def plot_comparison(all_metrics, save_dir=None):
        """Generate cross-scenario comparison bar chart."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = os.path.join("results")
        os.makedirs(save_dir, exist_ok=True)

        labels = []
        cav_rates = []
        rewards = []
        ttc_vals = []
        speeds = []
        travel_times = []
        vsl_e1, vsl_e2, vsl_e3 = [], [], []

        for m in all_metrics:
            labels.append(m['scenario'])
            cav_rates.append(m['cav_rate'])
            rewards.append(m['mean_reward'])
            ttc_vals.append(m['mean_ttc_violations'])
            speeds.append(m['mean_speed'])
            travel_times.append(m['mean_travel_time'])
            vsl_e1.append(m['avg_vsl_E1'])
            vsl_e2.append(m['avg_vsl_E2'])
            vsl_e3.append(m['avg_vsl_E3'])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))

        def bar(ax, vals, title, ylabel, fmt='.2f'):
            bars = ax.bar(labels, vals, color=colors)
            for bar_, v in zip(bars, vals):
                ax.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height(),
                        f'{v:{fmt}}', ha='center', va='bottom', fontsize=9)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis='y')

        bar(axes[0, 0], rewards, 'Mean Reward', 'Reward')
        bar(axes[0, 1], ttc_vals, 'TTC Violations (<3s)', 'Count')
        bar(axes[0, 2], speeds, 'Average Speed', 'm/s')

        # Travel time with free-flow reference
        axes[1, 0].bar(labels, travel_times, color=colors)
        axes[1, 0].axhline(y=90.0, color='r', linestyle='--', alpha=0.6, label='Free-flow (90s)')
        for i, v in enumerate(travel_times):
            axes[1, 0].text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
        axes[1, 0].set_title('Mean Travel Time (E1-E5)')
        axes[1, 0].set_ylabel('Seconds')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # VSL bars grouped
        x = np.arange(len(labels))
        width = 0.25
        axes[1, 1].bar(x - width, vsl_e1, width, label='E1', color='#e74c3c')
        axes[1, 1].bar(x, vsl_e2, width, label='E2', color='#2ecc71')
        axes[1, 1].bar(x + width, vsl_e3, width, label='E3', color='#3498db')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].set_title('Average VSL by Edge')
        axes[1, 1].set_ylabel('VSL (m/s)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # CAV rate info
        axes[1, 2].axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'scenario_comparison.png'), dpi=150)
        plt.close(fig)

        print(f"  -> Comparison plot saved to {save_dir}")


# ================================================================
# Standalone test
# ================================================================

if __name__ == "__main__":
    import config as cfg

    print("Testing SACVSLTrainer with a single episode (no training)...")
    trainer = SACVSLTrainer(cfg, "cav100", gui=False)

    # Run one evaluation episode to verify everything connects
    metrics = trainer.evaluate(num_episodes=1)
    print("\nTest complete!")
