"""
run.py — CLI entry point for SAC VSL training and evaluation.

Usage:
  python run.py --scenario cav100 --mode train --episodes 300
  python run.py --scenario all --mode train --episodes 300
  python run.py --scenario cav50 --mode eval --checkpoint checkpoints/sac_vsl_cav50_best.pt
  python run.py --scenario all --mode eval --checkpoint checkpoints/

Python environment: D:/anaconda/envs/LLM_Classification/python.exe
"""

import os
import sys
import argparse

# Ensure SUMO tools are available
os.environ["SUMO_HOME"] = "D:/sumo-win64-1.26.0/sumo-1.26.0"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import config
from sac_vsl import SACVSLTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC-based Variable Speed Limit Control for Highway Work Zone"
    )
    parser.add_argument(
        "--scenario", type=str, default="cav100",
        choices=["cav25", "cav50", "cav75", "cav100", "all"],
        help="CAV penetration scenario (default: cav100)"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "eval"],
        help="Mode: train or eval (default: train)"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of episodes (default: 300 for train, 10 for eval)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch SUMO with GUI (for debugging)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file or directory for eval/resume"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training (default: auto)"
    )
    return parser.parse_args()


def get_checkpoint_path(scenario, args):
    """Resolve checkpoint path from args."""
    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            # Look for best model in directory
            ckpt_file = f"sac_vsl_{scenario}_best.pt"
            path = os.path.join(args.checkpoint, ckpt_file)
            if os.path.exists(path):
                return path
            # Try final model
            ckpt_file = f"sac_vsl_{scenario}_final.pt"
            path = os.path.join(args.checkpoint, ckpt_file)
            if os.path.exists(path):
                return path
            print(f"Warning: no checkpoint found in {args.checkpoint} for {scenario}")
            return None
        else:
            return args.checkpoint
    return None


def main():
    args = parse_args()

    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios = list(config.SCENARIOS.keys())
    else:
        scenarios = [args.scenario]

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Set device if specified
    if args.device != "auto":
        import torch
        if args.device == "cpu":
            torch.set_default_device("cpu")
        # CUDA is default if available

    results = {}

    for scenario in scenarios:
        print(f"\n{'#'*60}")
        print(f"# Scenario: {scenario} (CAV {config.SCENARIOS[scenario][1]:.0%})")
        print(f"{'#'*60}")

        ckpt_path = get_checkpoint_path(scenario, args)

        # Create trainer
        trainer = SACVSLTrainer(
            config=config,
            scenario_name=scenario,
            gui=args.gui,
            checkpoint_path=ckpt_path if args.mode == "train" else ckpt_path,
        )

        if args.mode == "train":
            episodes = args.episodes if args.episodes else config.TRAIN_EPISODES
            trainer.train(num_episodes=episodes)

            # Plot training results
            trainer.plot_results()

            # Evaluate after training
            print("\n--- Post-training evaluation ---")
            metrics = trainer.evaluate(num_episodes=config.EVAL_EPISODES)
            results[scenario] = metrics

        elif args.mode == "eval":
            if ckpt_path is None:
                print(f"Warning: no checkpoint provided for {scenario}, using untrained agent.")
            episodes = args.episodes if args.episodes else config.EVAL_EPISODES
            metrics = trainer.evaluate(num_episodes=episodes)
            results[scenario] = metrics

    # Print comparison across scenarios
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("Cross-Scenario Comparison")
        print(f"{'='*70}")
        print(f"{'Scenario':>10} {'CAV%':>6} {'Avg Reward':>12} {'TTC Viol':>10} {'Avg Speed':>10} {'Avg TT':>10} {'VSL E1-3':>18}")
        print("-" * 70)
        for s, m in results.items():
            vsl_str = f"{m['avg_vsl_E1']:.1f}/{m['avg_vsl_E2']:.1f}/{m['avg_vsl_E3']:.1f}"
            print(f"{s:>10} {m['cav_rate']:6.0%} {m['mean_reward']:12.2f} "
                  f"{m['mean_ttc_violations']:10.1f} {m['mean_speed']:10.2f} "
                  f"{m['mean_travel_time']:10.1f} {vsl_str:>18}")

        # Plot comparison
        SACVSLTrainer.plot_comparison(list(results.values()))

    elif len(results) == 1:
        # Single scenario: save eval metrics as CSV for reference
        s, m = next(iter(results.items()))
        print(f"\n  === {s} Summary ===")
        for k, v in m.items():
            print(f"    {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
