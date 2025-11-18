#!/usr/bin/env python3
"""Build fixed-length (state, action) context windows from a simple .npz.

Expected input keys (minimally):
  - 'state':  (N, S) array
  - 'action': (N, A) array

Optional input keys (any one helps define episode boundaries):
  - 'episode': (N,) int array of episode ids (contiguous blocks per id)
  - 'done':    (N,) bool/0-1 array; True marks end of episode

If neither 'episode' nor 'done' is present, all data is treated as a single episode.

Output (.npz):
  - state_ctx:   (N, T, S)
  - action_ctx:  (N, T, A)
  - target_action: (N, A)  # action at t
  - valid_mask:  (N, T) bool  # True where context positions are valid
  - episode:     (N,) int
  - timestep:    (N,) int  # per-episode timestep
  - metadata:    JSON string with schema info
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _stack_windows(series: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Right-aligned, zero-padded windows + validity mask."""
    n_steps, feat_dim = series.shape
    windows = np.zeros((n_steps, length, feat_dim), dtype=series.dtype)
    mask = np.zeros((n_steps, length), dtype=bool)
    for t in range(n_steps):
        start = max(0, t - length + 1)
        span = series[start : t + 1]
        windows[t, -span.shape[0] :] = span
        mask[t, -span.shape[0] :] = True
    return windows, mask


def _split_blocks_from_episode(ep: np.ndarray) -> List[np.ndarray]:
    """Contiguous index blocks per episode id."""
    change_points = np.nonzero(np.diff(ep) != 0)[0] + 1
    blocks = np.split(np.arange(len(ep)), change_points)
    return [blk for blk in blocks if blk.size > 0]


def _split_blocks_from_done(done: np.ndarray) -> List[np.ndarray]:
    """Contiguous index blocks using done=True as episode end."""
    done = done.astype(bool)
    idxs = np.arange(len(done))
    # episode ends *at* i if done[i] == True
    ends = np.where(done)[0]
    starts = np.concatenate(([0], ends + 1))
    stops = np.concatenate((ends + 1, [len(done)]))
    blocks = [idxs[s:e] for s, e in zip(starts, stops) if e > s]
    return blocks


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Make sure array is (N, D) even if D == 1."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def build_ctx_from_npz(
    input_path: Path,
    output_path: Path,
    context: int,
    state_key: str,
    action_key: str,
    episode_key: str | None,
    done_key: str | None,
) -> None:
    data = np.load(input_path, allow_pickle=True)

    if state_key not in data.files or action_key not in data.files:
        raise KeyError(
            f"Input file must contain '{state_key}' and '{action_key}'. "
            f"Found: {list(data.files)}"
        )

    state = _ensure_2d(np.asarray(data[state_key], dtype=np.float32))
    action = _ensure_2d(np.asarray(data[action_key], dtype=np.float32))

    if state.shape[0] != action.shape[0]:
        raise ValueError(f"State and action length mismatch: {state.shape[0]} vs {action.shape[0]}")

    N = state.shape[0]

    # Resolve episodes
    if episode_key and episode_key in data.files:
        episodes = np.asarray(data[episode_key]).astype(int)
        if episodes.shape[0] != N:
            raise ValueError("episode vector length mismatch.")
        blocks = _split_blocks_from_episode(episodes)
        ep_ids = episodes.copy()
    elif done_key and done_key in data.files:
        done = np.asarray(data[done_key]).astype(bool)
        if done.shape[0] != N:
            raise ValueError("done vector length mismatch.")
        blocks = _split_blocks_from_done(done)
        # Assign episode ids incrementally
        ep_ids = np.empty(N, dtype=int)
        for eid, blk in enumerate(blocks):
            ep_ids[blk] = eid
    else:
        # Single episode
        blocks = [np.arange(N)]
        ep_ids = np.zeros(N, dtype=int)

    # Compute per-episode timesteps
    timesteps = np.empty(N, dtype=int)
    for blk in blocks:
        # timesteps start at 0 within each episode
        t = np.arange(len(blk), dtype=int)
        timesteps[blk] = t

    # Build windows episode-by-episode (prevents leakage across boundaries)
    st_ctx_list, ac_ctx_list, mask_list, tgt_list, ep_list, ts_list = [], [], [], [], [], []
    for blk in blocks:
        s_blk = state[blk]
        a_blk = action[blk]

        s_win, m_win = _stack_windows(s_blk, context)
        a_win, _ = _stack_windows(a_blk, context)

        st_ctx_list.append(s_win)
        ac_ctx_list.append(a_win)
        mask_list.append(m_win)
        tgt_list.append(a_blk)          # supervise action at t
        ep_list.append(ep_ids[blk])
        ts_list.append(timesteps[blk])

    state_ctx = np.concatenate(st_ctx_list, axis=0)
    action_ctx = np.concatenate(ac_ctx_list, axis=0)
    valid_mask = np.concatenate(mask_list, axis=0)
    target_action = np.concatenate(tgt_list, axis=0)
    episode_vec = np.concatenate(ep_list, axis=0)
    timestep_vec = np.concatenate(ts_list, axis=0)

    meta = {
        "context_length": int(context),
        "state_key": state_key,
        "action_key": action_key,
        "episode_source": (episode_key if (episode_key and episode_key in data.files)
                           else (done_key if (done_key and done_key in data.files)
                                 else "single_episode")),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "num_samples": int(state_ctx.shape[0]),
    }

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        state_ctx=state_ctx,
        action_ctx=action_ctx,
        target_action=target_action,
        valid_mask=valid_mask,
        episode=episode_vec,
        timestep=timestep_vec,
        metadata=json.dumps(meta),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create (state, action) context windows from a simple .npz")
    p.add_argument("--input", type=Path, required=True, help="Input .npz with 'state' and 'action'")
    p.add_argument("--output", type=Path, required=True, help="Output .npz path")
    p.add_argument("--context", type=int, default=16, help="Window length (tokens per side)")
    p.add_argument("--state-key", default="state", help="Key for state array in input npz")
    p.add_argument("--action-key", default="action", help="Key for action array in input npz")
    p.add_argument("--episode-key", default=None, help="Optional key for episode ids in input npz")
    p.add_argument("--done-key", default=None, help="Optional key for done/terminal flags in input npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_ctx_from_npz(
        input_path=args.input,
        output_path=args.output,
        context=args.context,
        state_key=args.state_key,
        action_key=args.action_key,
        episode_key=args.episode_key,
        done_key=args.done_key,
    )


if __name__ == "__main__":
    main()
