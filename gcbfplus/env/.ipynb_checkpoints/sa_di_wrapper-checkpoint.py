
# gcbfplus/env/sa_di_wrapper.py
from __future__ import annotations
import numpy as np
import jax.random as jr
from typing import Dict, Tuple, Optional, Any

class SingleAgentDIEnv:
    """Black-box, single-agent wrapper around a GCBF+ DoubleIntegrator env.

    Goals:
      - Single-agent Gym-like API for model-free training
      - Keep *black-box* predictive utilities (clone/rollout_sequence) for GCBF+-style labeling & teacher
      - Expose car_radius, comm_radius, n_rays
      - Include LiDAR distances in observations

    Observation dict:
      {
        'state': [x,y,vx,vy],
        'goal':  [gx,gy,gvx,gvy],
        'state_goal': [x,y,vx,vy, gx-x, gy-y],
        'lidar': [n_rays]  (distances agent→hit, normalized if normalize_lidar=True),
        'full_obs': flat vector of selected pieces (see __init__ flags),
        'action_low': [2], 'action_high': [2],
        'car_radius': float, 'comm_radius': float, 'n_rays': int
      }
    """
    def __init__(self, env_m,
                 include_velocity: bool = True,
                 normalize_lidar: bool = True,
                 full_obs_keys: Tuple[str, ...] = ("state_goal", "lidar")):
        assert getattr(env_m, 'num_agents', 1) == 1, "Please instantiate the inner env with num_agents=1"
        self._env = env_m
        self._graph = None
        self._key = None
        self._include_velocity = include_velocity
        self._normalize_lidar = normalize_lidar
        # Build flat full_obs from these pieces in this order
        # Valid keys: 'state' | 'goal' | 'state_goal' | 'lidar'
        self._full_obs_keys = tuple(full_obs_keys)

    # ---------- parameter helpers ----------
    def _param(self, name, default):
        for key in ("params", "_params", "PARAMS"):
            if hasattr(self._env, key):
                p = getattr(self._env, key)
                try:
                    if isinstance(p, dict) and name in p:
                        return p[name]
                    return p.get(name, default)
                except Exception:
                    pass
        return default

    @property
    def car_radius(self) -> float:
        return float(self._param("car_radius", 0.05))
    @property
    def comm_radius(self) -> float:
        return float(self._param("comm_radius", 0.5))
    @property
    def n_rays(self) -> int:
        return int(self._param("n_rays", 32))

    # ---------- convenience properties ----------
    @property
    def action_low(self):
        low, _ = self._env.action_lim()
        return np.array(low, dtype=np.float32).reshape(-1)
    @property
    def action_high(self):
        _, high = self._env.action_lim()
        return np.array(high, dtype=np.float32).reshape(-1)
    @property
    def dt(self) -> float:
        return float(self._env.dt)
    @property
    def area_size(self) -> float:
        return float(self._env.area_size)

    # ---------- internals ----------
    def _extract_agent_goal(self, graph) -> Tuple[np.ndarray, np.ndarray]:
        agent = np.array(graph.env_states.agent)[0]  # [x,y,vx,vy]
        goal  = np.array(graph.env_states.goal)[0]   # [gx,gy,gvx,gvy]
        return agent.astype(np.float32), goal.astype(np.float32)

    def _extract_lidar(self, graph) -> np.ndarray:
        # lidar nodes are type_idx=2 with length = n_rays * num_agents (= n_rays since single agent)
        n = self.n_rays
        lidar_nodes = np.array(graph.type_states(type_idx=2, n_type=n))  # [n, 4]; first two are positions
        agent_pos = np.array(graph.env_states.agent)[0, :2]
        distances = np.linalg.norm(lidar_nodes[:, :2] - agent_pos[None, :], axis=1)
        if self._normalize_lidar and self.comm_radius > 0:
            distances = distances / self.comm_radius
        return distances.astype(np.float32)

    def _build_full_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for key in self._full_obs_keys:
            if key == "state_goal" and self._include_velocity is False:
                # if velocity excluded, rebuild state_goal without v
                s = obs["state"]
                sg = np.array([s[0], s[1], obs["goal"][0]-s[0], obs["goal"][1]-s[1]], dtype=np.float32)
                parts.append(sg)
            else:
                parts.append(obs[key].astype(np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _obs_from_graph(self, graph) -> Dict[str, np.ndarray]:
        s, g = self._extract_agent_goal(graph)
        dx_dy = g[:2] - s[:2]
        if not self._include_velocity:
            s = np.array([s[0], s[1], 0.0, 0.0], dtype=np.float32)  # mask velocity out, if you want strict partial obs
        lidar = self._extract_lidar(graph)
        obs = {
            "state": s,
            "goal": g,
            "state_goal": np.concatenate([s, dx_dy], axis=0).astype(np.float32),
            "lidar": lidar,
            "action_low": self.action_low.copy(),
            "action_high": self.action_high.copy(),
            "car_radius": float(self.car_radius),
            "comm_radius": float(self.comm_radius),
            "n_rays": int(self.n_rays),
        }
        obs["full_obs"] = self._build_full_obs(obs)
        return obs

    # ---------- public API ----------
    def reset(self, seed: Optional[int] = None, state: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._key = jr.PRNGKey(seed)
        self._graph = self._env.reset(self._key)
        if state is not None:
            self.set_state(state)
        return self._obs_from_graph(self._graph)

    def step(self, action: np.ndarray, get_eval_info: bool = True):
        action = np.asarray(action, dtype=np.float32).reshape(1, 2)
        graph_new, reward, cost, done, info = self._env.step(self._graph, action, get_eval_info=get_eval_info)
        self._graph = graph_new
        obs = self._obs_from_graph(self._graph)
        out_info = dict(info)
        # Cast scalars
        out_info["cost"] = float(np.array(out_info.get("cost", cost)))
        out_info["done"] = bool(np.array(out_info.get("done", done)))
        if "inside_obstacles" in out_info:
            try:
                out_info["inside_obstacles"] = bool(np.array(out_info["inside_obstacles"]))
            except Exception:
                pass
        return obs, float(np.array(reward)), bool(np.array(done)), out_info

    # Black-box state access for teacher/labeling (do not use in policy if you want strict model-free)
    def get_state(self) -> np.ndarray:
        assert self._graph is not None, "reset() first"
        s, _ = self._extract_agent_goal(self._graph)
        return s
    def set_state(self, state: np.ndarray):
        assert self._graph is not None, "reset() first"
        state = np.asarray(state, dtype=np.float32).reshape(1, 4)
        es = self._graph.env_states
        EnvStateT = type(es)
        new_env_state = EnvStateT(agent=state, goal=np.array(es.goal, dtype=np.float32), obstacle=es.obstacle)
        self._graph = self._env.get_graph(new_env_state)
    def clone(self) -> "SingleAgentDIEnv":
        new = SingleAgentDIEnv(self._env,
                               include_velocity=self._include_velocity,
                               normalize_lidar=self._normalize_lidar,
                               full_obs_keys=self._full_obs_keys)
        assert self._graph is not None, "reset() first"
        es = self._graph.env_states
        EnvStateT = type(es)
        agent = np.array(es.agent, dtype=np.float32)
        goal  = np.array(es.goal,  dtype=np.float32)
        obstacle = es.obstacle
        new_env_state = EnvStateT(agent=agent, goal=goal, obstacle=obstacle)
        new._graph = self._env.get_graph(new_env_state)
        new._key = self._key
        return new

    def is_geom_safe(self, state: Optional[np.ndarray] = None) -> bool:
        if state is None:
            graph = self._graph
        else:
            es = self._graph.env_states
            EnvStateT = type(es)
            agent = np.asarray(state, dtype=np.float32).reshape(1,4)
            goal  = np.array(es.goal, dtype=np.float32)
            obstacle = es.obstacle
            graph = self._env.get_graph(EnvStateT(agent=agent, goal=goal, obstacle=obstacle))
        cost = float(np.array(self._env.get_cost(graph)))
        return cost == 0.0

    def rollout_sequence(self, actions: np.ndarray, early_stop_on_violation: bool = False):
        sim = self.clone()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        H = actions.shape[0]
        states = [sim.get_state().copy()]
        rewards, infos = [], []
        for k in range(H):
            _obs, rew, done, info = sim.step(actions[k], get_eval_info=True)
            states.append(sim.get_state().copy())
            rewards.append(rew)
            infos.append(info)
            io = bool(np.array(info.get('inside_obstacles', False))) if 'inside_obstacles' in info else False
            if early_stop_on_violation and (io or info.get('done', False)):
                break
        return np.stack(states, axis=0), np.array(rewards, dtype=np.float32), infos

    # Model-input convenience
    def feature_vector(self, obs: Optional[Dict[str, np.ndarray]] = None,
                       include_bounds: bool = True, include_radii: bool = True,
                       use_full_obs: bool = True) -> np.ndarray:
        if obs is None:
            obs = self._obs_from_graph(self._graph)
        base = obs["full_obs"] if use_full_obs else obs["state_goal"]
        parts = [base.astype(np.float32)]
        if include_bounds:
            parts += [self.action_low, self.action_high]
        if include_radii:
            parts += [np.array([self.car_radius, self.comm_radius], dtype=np.float32)]
        return np.concatenate(parts, axis=0).astype(np.float32)
# # sa_di_wrapper.py
# # Single-Agent wrapper for GCBF+ DoubleIntegrator environment.
# # Keeps the simulator as a black box while exposing a Gym-like API and utilities
# # needed for predictive safety (MPC-style shooting) and GCBF-style labeling.
# #
# # Usage:
# #   from double_integrator import DoubleIntegrator
# #   env_m = DoubleIntegrator(num_agents=1, area_size=6.0)
# #   env = SingleAgentDIEnv(env_m)
# #   obs = env.reset(seed=0)
# #   obs, reward, done, info = env.step(action)
# #
# # The wrapper assumes the wrapped env uses the same API as GCBF+:
# # - reset(key) -> GraphsTuple
# # - step(graph, action, get_eval_info: bool) -> (GraphsTuple, reward, cost, done, info)
# # - get_graph(EnvState) -> GraphsTuple, where EnvState contains fields (agent, goal, obstacle)
# # - get_cost(graph) -> scalar cost (collision indicator averaged over agents)
# #
# # This file is self-contained and can live anywhere in your project as long as you pass a valid env instance.

# from __future__ import annotations
# import copy
# import numpy as np
# import jax
# import jax.numpy as jnp
# import jax.random as jr
# from typing import Dict, Tuple, Optional, Any

# class SingleAgentDIEnv:
#     """Black-box, single-agent wrapper around a GCBF+ DoubleIntegrator env.

#     Exposes:
#       - reset(seed: Optional[int], state: Optional[np.ndarray]) -> obs
#       - step(action: np.ndarray) -> (obs, reward, done, info)
#       - get_state() / set_state(state)
#       - clone()  (deepcopy environment + current graph)
#       - is_geom_safe(state=None, clearance: float=0.0) -> bool
#       - rollout_sequence(actions: np.ndarray, early_stop_on_violation: bool = False)

#     Observation returned by reset/step is a dict with:
#       {'state': np.ndarray [4], 'goal': np.ndarray [4], 'state_goal': np.ndarray [6]}
#       where state_goal concatenates [x, y, vx, vy, goal_x - x, goal_y - y].
#     """

#     def __init__(self, env_m):
#         assert getattr(env_m, 'num_agents', 1) == 1, "Please instantiate the inner env with num_agents=1"
#         self._env = env_m
#         self._graph = None  # GraphsTuple
#         self._key = None

#     # -------------- Helpers --------------
#     def _extract_agent_goal(self, graph) -> Tuple[np.ndarray, np.ndarray]:
#         # graph.env_states.agent: shape (1, 4); graph.env_states.goal: shape (1, 4)
#         agent = np.array(graph.env_states.agent)[0]  # [x,y,vx,vy]
#         goal  = np.array(graph.env_states.goal)[0]   # [gx,gy,gvx,gvy] (vels often zero)
#         return agent.astype(np.float32), goal.astype(np.float32)

#     def _obs_from_graph(self, graph) -> Dict[str, np.ndarray]:
#         s, g = self._extract_agent_goal(graph)
#         dx_dy = g[:2] - s[:2]
#         obs = {
#             "state": s,                   # [x, y, vx, vy]
#             "goal": g,                    # [gx, gy, gvx, gvy]
#             "state_goal": np.concatenate([s, dx_dy], axis=0).astype(np.float32)  # [x,y,vx,vy, gx-x, gy-y]
#         }
#         return obs

#     # -------------- Public API --------------
#     def reset(self, seed: Optional[int] = None, state: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
#         if seed is None:
#             seed = np.random.randint(0, 2**31 - 1)
#         self._key = jr.PRNGKey(seed)
#         self._graph = self._env.reset(self._key)
#         if state is not None:
#             self.set_state(state)
#         return self._obs_from_graph(self._graph)

#     def step(self, action: np.ndarray, get_eval_info: bool = True) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
#         action = np.asarray(action, dtype=np.float32).reshape(1, 2)
#         graph_new, reward, cost, done, info = self._env.step(self._graph, action, get_eval_info=get_eval_info)
#         self._graph = graph_new
#         obs = self._obs_from_graph(self._graph)
#         # Standardize outputs
#         out_info = dict(info)
#         out_info.setdefault("cost", float(np.array(cost)))
#         out_info.setdefault("done", bool(np.array(done)))
#         return obs, float(np.array(reward)), bool(np.array(done)), out_info

#     def get_state(self) -> np.ndarray:
#         assert self._graph is not None, "Call reset() first"
#         s, _ = self._extract_agent_goal(self._graph)
#         return s

#     def set_state(self, state: np.ndarray):
#         assert self._graph is not None, "Call reset() first"
#         state = np.asarray(state, dtype=np.float32).reshape(1, 4)
#         g = self._graph.env_states.goal
#         obs_struct = self._graph.env_states.obstacle
#         # Rebuild EnvState via the env's typed container (DoubleIntegrator.EnvState)
#         EnvState = type(self._env.EnvState) if not hasattr(self._env, 'EnvState') else self._env.EnvState
#         new_env_state = EnvState(agent=state, goal=g, obstacle=obs_struct)
#         self._graph = self._env.get_graph(new_env_state)

#     def clone(self) -> "SingleAgentDIEnv":
#         # Deep copy env and current graph for safe shooting
#         new_env = copy.deepcopy(self._env)
#         new = SingleAgentDIEnv(new_env)
#         new._graph = copy.deepcopy(self._graph)
#         new._key = copy.deepcopy(self._key)
#         return new

#     def is_geom_safe(self, state: Optional[np.ndarray] = None, clearance: float = 0.0) -> bool:
#         """Geometric safety based on the wrapped env's cost (obstacle collisions).
#         For a single agent, env.get_cost(graph) > 0 indicates obstacle or (hypothetical) agent collision.
#         """
#         if state is not None:
#             cur = self.clone()
#             cur.set_state(state)
#             cost = float(np.array(cur._env.get_cost(cur._graph)))
#         else:
#             assert self._graph is not None, "Call reset() first"
#             cost = float(np.array(self._env.get_cost(self._graph)))
#         return cost == 0.0

#     # ---------- Rollout utilities (for teacher & labeling) ----------
#     def rollout_sequence(self, actions: np.ndarray, early_stop_on_violation: bool = False):
#         """Roll out a sequence from the *current* state on a cloned simulator.

#         Returns:
#           states: np.ndarray [H+1, 4]
#           rewards: np.ndarray [H]
#           infos: list of dicts (per step)
#         """
#         sim = self.clone()
#         actions = np.asarray(actions, dtype=np.float32)
#         if actions.ndim == 1:
#             actions = actions[None, :]  # [1,2] -> [1,2]
#         H = actions.shape[0]
#         states = [sim.get_state().copy()]
#         rewards = []
#         infos = []
#         for k in range(H):
#             obs, rew, done, info = sim.step(actions[k], get_eval_info=True)
#             states.append(sim.get_state().copy())
#             rewards.append(rew)
#             infos.append(info)
#             if early_stop_on_violation and (info.get('inside_obstacles', False) or info.get('done', False)):
#                 break
#         return np.stack(states, axis=0), np.array(rewards, dtype=np.float32), infos

#     # ---------- Convenience: bounds ----------
#     @property
#     def action_low(self) -> np.ndarray:
#         low, _ = self._env.action_lim()
#         return np.array(low, dtype=np.float32).reshape(-1)

#     @property
#     def action_high(self) -> np.ndarray:
#         _, high = self._env.action_lim()
#         return np.array(high, dtype=np.float32).reshape(-1)

#     @property
#     def dt(self) -> float:
#         return float(self._env.dt)

#     @property
#     def area_size(self) -> float:
#         return float(self._env.area_size)

#     def _param(self, name, default):
#         # Robustly read from env.params, env._params, or class-level PARAMS
#         for key in ("params", "_params", "PARAMS"):
#             if hasattr(self._env, key):
#                 p = getattr(self._env, key)
#                 try:
#                     if isinstance(p, dict) and name in p:
#                         return p[name]
#                     # handle dataclass-like / Mapping
#                     return p.get(name, default)
#                 except Exception:
#                     pass
#         return default

#     @property
#     def car_radius(self) -> float:
#         return float(self._param("car_radius", 0.05))

#     @property
#     def comm_radius(self) -> float:
#         return float(self._param("comm_radius", 0.5))

#     @property
#     def n_rays(self) -> int:
#         return int(self._param("n_rays", 32))

#     @property
#     def action_low(self):
#         low, _ = self._env.action_lim()
#         import numpy as _np
#         return _np.array(low, dtype=_np.float32).reshape(-1)

#     @property
#     def action_high(self):
#         _, high = self._env.action_lim()
#         import numpy as _np
#         return _np.array(high, dtype=_np.float32).reshape(-1)

#     @property
#     def dt(self) -> float:
#         return float(self._env.dt)

#     @property
#     def area_size(self) -> float:
#         return float(self._env.area_size)

