#!/usr/bin/env python3
"""Collect offline rollouts for training the PACT transformer on DoubleIntegrator."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect (state, action, safety) dataset for PACT training")
    parser.add_argument("--episodes", type=int, default=128, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=256, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--output", type=Path, default=Path("pact_dataset.npz"), help="Output .npz path")
    parser.add_argument("--area-size", type=float, default=6.0, help="Environment square size")
    parser.add_argument("--dt", type=float, default=0.03, help="Simulation time step")
    parser.add_argument("--max-travel", type=float, default=None, help="Maximum travel distance for targets")
    parser.add_argument(
        "--perturb-prob",
        type=float,
        default=0.2,
        help="Probability of adding exploratory noise to the nominal action",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.1,
        help="Std-dev of Gaussian noise added to the nominal action when perturbing",
    )
    parser.add_argument(
        "--normalize-lidar",
        action="store_true",
        default=True,
        help="Store LiDAR distances normalized by comm radius",
    )
    parser.add_argument(
        "--no-normalize-lidar",
        dest="normalize_lidar",
        action="store_false",
        help="Disable LiDAR normalization",
    )
    parser.add_argument(
        "--full-obs",
        nargs="*",
        default=["state_goal", "lidar"],
        help="Keys to concatenate for obs['full_obs']",
    )
    parser.add_argument(
        "--mode-excitation",
        action="store_true",
        help="Cycle additive actions through all sign quadrants (++,+-,-+,--)",
    )
    parser.add_argument(
        "--excitation-interval",
        type=int,
        default=8,
        help="Steps between forced excitation injections when enabled",
    )
    parser.add_argument(
        "--excitation-scale",
        type=float,
        default=0.5,
        help="Fraction of action range used for mode excitation deltas",
    )
    parser.add_argument(
        "--n-obs",
        "--n_obs",
        dest="n_obs",
        type=int,
        default=None,
        help="Override number of rectangular obstacles (None keeps env default)",
    )  # Accept --n_obs to match user invocation preference.
    parser.add_argument(
        "--obs-len-min",
        type=float,
        default=None,
        help="Override minimum obstacle side length",
    )
    parser.add_argument(
        "--obs-len-max",
        type=float,
        default=None,
        help="Override maximum obstacle side length",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env_params = None
    if any(v is not None for v in (args.n_obs, args.obs_len_min, args.obs_len_max)):
        env_params = dict(DoubleIntegrator.PARAMS)
        if args.n_obs is not None:
            env_params["n_obs"] = int(args.n_obs)
        if args.obs_len_min is not None or args.obs_len_max is not None:
            lo = args.obs_len_min if args.obs_len_min is not None else env_params["obs_len_range"][0]
            hi = args.obs_len_max if args.obs_len_max is not None else env_params["obs_len_range"][1]
            env_params["obs_len_range"] = [float(lo), float(hi)]

    env_m = DoubleIntegrator(
        num_agents=1,
        area_size=args.area_size,
        max_step=args.max_steps,
        max_travel=args.max_travel,
        dt=args.dt,
        params=env_params,
    )
    env = SingleAgentDIEnv(
        env_m,
        include_velocity=True,
        normalize_lidar=args.normalize_lidar,
        full_obs_keys=tuple(args.full_obs),
    )

    records = {
        "episode": [],
        "timestep": [],
        "state": [],
        "goal": [],
        "lidar": [],
        "full_obs": [],
        "nominal_action": [],
        "taken_action": [],
        "perturbed": [],
        "mode_excited": [],
        "excitation_direction": [],
        "reward": [],
        "cost": [],
        "safe_mask": [],
        "unsafe_mask": [],
        "collision_mask": [],
        "finish_mask": [],
        "next_state": [],
        "next_full_obs": [],
        "next_safe_mask": [],
    }

    step_counter = 0
    mode_cycle = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    mode_index = 0
    for episode in range(args.episodes):
        obs = env.reset(seed=args.seed + episode)
        for t in range(args.max_steps):
            state_vec = obs["state"].astype(np.float32)
            goal_vec = obs["goal"].astype(np.float32)
            lidar_vec = obs["lidar"].astype(np.float32)
            full_obs_vec = obs["full_obs"].astype(np.float32)
            current_state = env.get_state().copy()

            safe_flag = env.safe_mask(current_state)
            unsafe_flag = env.unsafe_mask(current_state)
            collision_flag = env.collision_mask(current_state)
            finish_flag = env.finish_mask(current_state)

            nominal_action = env.nominal_action(current_state)
            taken_action = nominal_action.copy()
            perturbed = False
            mode_excited = False
            excitation_direction = np.zeros_like(taken_action)
            if rng.random() < args.perturb_prob:
                taken_action += rng.normal(scale=args.perturb_scale, size=nominal_action.shape)
                taken_action = np.clip(taken_action, env.action_low, env.action_high)
                perturbed = True

            if args.mode_excitation and (t % max(args.excitation_interval, 1) == 0):
                direction = mode_cycle[mode_index]
                mode_index = (mode_index + 1) % len(mode_cycle)
                action_range = (env.action_high - env.action_low) * 0.5
                excitation = args.excitation_scale * action_range * direction
                taken_action += excitation
                taken_action = np.clip(taken_action, env.action_low, env.action_high)
                mode_excited = True
                excitation_direction = direction
                perturbed = True

            next_obs, reward, done, info = env.step(taken_action)
            next_state = env.get_state().copy()
            next_full_obs = next_obs["full_obs"].astype(np.float32)
            next_safe_flag = env.safe_mask()
            cost_value = info.get("cost")
            if cost_value is None:
                cost_value = env.get_cost()

            records["episode"].append(episode)
            records["timestep"].append(t)
            records["state"].append(state_vec)
            records["goal"].append(goal_vec)
            records["lidar"].append(lidar_vec)
            records["full_obs"].append(full_obs_vec)
            records["nominal_action"].append(nominal_action.astype(np.float32))
            records["taken_action"].append(taken_action.astype(np.float32))
            records["perturbed"].append(perturbed)
            records["mode_excited"].append(mode_excited)
            records["excitation_direction"].append(excitation_direction.astype(np.float32))
            records["reward"].append(np.float32(reward))
            records["cost"].append(np.float32(cost_value))
            records["safe_mask"].append(safe_flag)
            records["unsafe_mask"].append(unsafe_flag)
            records["collision_mask"].append(collision_flag)
            records["finish_mask"].append(finish_flag)
            records["next_state"].append(next_state.astype(np.float32))
            records["next_full_obs"].append(next_full_obs)
            records["next_safe_mask"].append(next_safe_flag)

            obs = next_obs
            step_counter += 1
            if done:
                break

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {key: np.array(value) for key, value in records.items()}
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "perturb_prob": args.perturb_prob,
        "perturb_scale": args.perturb_scale,
        "normalize_lidar": args.normalize_lidar,
        "full_obs_keys": list(args.full_obs),
        "area_size": args.area_size,
        "dt": args.dt,
        "max_travel": args.max_travel,
        "n_obs": env_m.params["n_obs"],
        "obs_len_range": list(env_m.params["obs_len_range"]),
        "total_steps": step_counter,
    }
    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(output_path, **arrays)
    print(f"Saved {step_counter} transitions to {output_path}")


if __name__ == "__main__":
    main()