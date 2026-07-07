"""
CLI entry point for SAC VSL training and evaluation.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        help="CAV penetration scenario",
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of episodes",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch SUMO with GUI",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file or checkpoint directory",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training",
    )
    parser.add_argument(
        "--parallel-workers", type=int, default=4,
        help="Maximum workers when --scenario all is used",
    )
    return parser.parse_args()


def get_checkpoint_path(scenario, checkpoint):
    if checkpoint:
        if os.path.isdir(checkpoint):
            for name in (f"sac_vsl_{scenario}_best.pt", f"sac_vsl_{scenario}_final.pt"):
                path = os.path.join(checkpoint, name)
                if os.path.exists(path):
                    return path
            print(f"Warning: no checkpoint found in {checkpoint} for {scenario}")
            return None
        return checkpoint
    return None


def run_one_scenario(scenario, args_dict):
    import config as worker_config
    from sac_vsl import SACVSLTrainer

    if args_dict["device"] != "auto":
        import torch
        if args_dict["device"] == "cpu":
            torch.set_default_device("cpu")

    os.makedirs(worker_config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = get_checkpoint_path(scenario, args_dict["checkpoint"])
    trainer = SACVSLTrainer(
        config=worker_config,
        scenario_name=scenario,
        gui=args_dict["gui"],
        checkpoint_path=ckpt_path,
    )

    if args_dict["mode"] == "train":
        episodes = args_dict["episodes"] if args_dict["episodes"] else worker_config.TRAIN_EPISODES
        trainer.train(num_episodes=episodes)
        trainer.plot_results()
        metrics = trainer.evaluate(num_episodes=worker_config.EVAL_EPISODES)
    else:
        if ckpt_path is None:
            print(f"Warning: no checkpoint provided for {scenario}, using untrained agent.")
        episodes = args_dict["episodes"] if args_dict["episodes"] else worker_config.EVAL_EPISODES
        metrics = trainer.evaluate(num_episodes=episodes)

    return scenario, metrics


def print_comparison(results):
    if len(results) <= 1:
        scenario, metrics = next(iter(results.items()))
        print(f"\n  === {scenario} Summary ===")
        SACVSLTrainer._print_eval_summary("RL", metrics["rl"])
        SACVSLTrainer._print_eval_summary("Baseline 80km/h", metrics["baseline"])
        return

    print(f"\n{'=' * 110}")
    print("Cross-Scenario Comparison")
    print(f"{'=' * 110}")
    print(f"{'Scenario':>10} {'CAV%':>6} "
          f"{'RL TT':>12} {'Base TT':>12} "
          f"{'RL CO2':>12} {'Base CO2':>12} "
          f"{'RL TTC':>10} {'Base TTC':>10} "
          f"{'RL E4 Spd':>10} {'Base E4 Spd':>12}")
    print("-" * 110)
    for scenario, metrics in results.items():
        rl = metrics["rl"]
        baseline = metrics["baseline"]
        print(f"{scenario:>10} {metrics['cav_rate']:6.0%} "
              f"{rl['total_travel_time']:12.1f} {baseline['total_travel_time']:12.1f} "
              f"{rl['co2_emission']:12.1f} {baseline['co2_emission']:12.1f} "
              f"{rl['ttc_total']:10.1f} {baseline['ttc_total']:10.1f} "
              f"{rl['e4_mean_speed']:10.2f} {baseline['e4_mean_speed']:12.2f}")

    SACVSLTrainer.plot_comparison(list(results.values()))


def main():
    args = parse_args()
    scenarios = list(config.SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    args_dict = vars(args)
    results = {}

    if len(scenarios) > 1 and not args.gui:
        workers = min(args.parallel_workers, len(scenarios))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(run_one_scenario, scenario, args_dict): scenario
                for scenario in scenarios
            }
            for future in as_completed(future_map):
                scenario, metrics = future.result()
                results[scenario] = metrics
    else:
        for scenario in scenarios:
            scenario, metrics = run_one_scenario(scenario, args_dict)
            results[scenario] = metrics

    ordered_results = {scenario: results[scenario] for scenario in scenarios if scenario in results}
    print_comparison(ordered_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
