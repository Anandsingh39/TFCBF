
# # gcbfplus/env/sample.py
# import numpy as np
# from .double_integrator import DoubleIntegrator
# from .sa_di_wrapper import SingleAgentDIEnv

# def main():
#     base = DoubleIntegrator(num_agents=1, area_size=6.0)
#     env = SingleAgentDIEnv(base, include_velocity=True, normalize_lidar=True,
#                            full_obs_keys=("state_goal","lidar"))
#     obs = env.reset(seed=0)
#     print("car_radius:", env.car_radius, "comm_radius:", env.comm_radius, "n_rays:", env.n_rays)
#     print("obs keys:", list(obs.keys()))
#     print("full_obs shape:", obs["full_obs"])
#     print("lidar shape:", obs["lidar"].shape, "first few:", obs["lidar"][:5])

#     # one step
#     a0 = np.array([0.1, -0.1], dtype=np.float32)
#     obs, reward, done, info = env.step(a0)
#     print("step reward:", reward, "done:", done, "inside_obstacles:", info.get("inside_obstacles"))

#     # rollout_sequence demo (for teacher/labeling)
#     H = 8
#     cand = np.tile(a0, (H, 1))
#     states, rewards, infos = env.rollout_sequence(cand, early_stop_on_violation=True)
#     print("rollout states shape:", states.shape)

#     # feature vector for your model
#     fv = env.feature_vector(obs)
#     print("feature_vector shape:", fv.shape)

# if __name__ == "__main__":
#     main()

# gcbfplus/env/plot_from_init.py
import numpy as np
import matplotlib.pyplot as plt

from . import make_env                         # factory from __init__.py
from .sa_di_wrapper import SingleAgentDIEnv    # your single-agent wrapper
from .plot import get_obs_collection, plot_graph  # official plotting utils

def main(seed=0, steps=2000, include_velocity=True):
    # 1) Build a single-agent DI env with exactly 4 obstacles (rectangles)
    base = make_env(
        env_id="double_integrator",
        num_agents=1,
        area_size=4.0,
        num_obs=16,   # ← four rectangles
        n_rays=32,
    )
    env = SingleAgentDIEnv(
        base,
        include_velocity=include_velocity,
        normalize_lidar=True,
        full_obs_keys=("state_goal", "lidar"),
    )

    # 2) Reset & inspect input features (this is what your model consumes)
    obs = env.reset(seed=seed)
    print("car_radius:", env.car_radius, "comm_radius:", env.comm_radius, "n_rays:", env.n_rays)
    print("full_obs dim:", obs["full_obs"])
    print("first 10 of full_obs:", np.round(obs["full_obs"][:10], 4), obs["full_obs"].shape[0])

    # Optional: include bounds+radii in a meta feature vector
    fv = env.feature_vector(obs, include_bounds=True, include_radii=True, use_full_obs=True)
    print("=================================================")
    print("fv", fv, fv.shape[0])
    print("feature_vector dim (full_obs + bounds + radii):", fv.shape[0])

    # 3) Roll out a short PD-like controller to make a path
    traj = [env.get_state().copy()]
    for _ in range(steps):
        s = env.get_state(); p, v = s[:2], s[2:]
        g = obs["goal"][:2]
        a = 1.4*(g - p) - 0.5*v
        obs, _, _, _ = env.step(a)
        traj.append(env.get_state().copy())
    traj = np.stack(traj)
    print("traj", len(traj))
    print(steps)

    # 4) Plot with official helpers (no custom obstacle code)
    fig, ax = plt.subplots(figsize=(6, 6))
    L = base.area_size
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.25)

    # Obstacles: use plot.get_obs_collection (works with Rectangle/Cuboid/Sphere)  :contentReference[oaicite:2]{index=2}
    ax.add_collection(get_obs_collection(env._graph.env_states.obstacle, color="#8a0000", alpha=0.8))

    # Agent trajectory
    ax.plot(traj[:, 0], traj[:, 1], lw=2, label="trajectory")
    ax.scatter(traj[0, 0], traj[0, 1], c="k", s=60, label="start")

    # Agent & goal markers using plot_graph (simple circles)  :contentReference[oaicite:3]{index=3}
    r = base.params.get("car_radius", 0.05)
    agent_xy = traj[-1, :2][None, :]
    goal_xy  = env._graph.env_states.goal[:, :2]
    plot_graph(ax, agent_xy, radius=r, color="#0068ff", with_label=True)
    plot_graph(ax, goal_xy,  radius=r, color="#2fdd00", with_label=True)

    ax.legend(); ax.set_title("Single-Agent DI (4 Rectangular Obstacles)")
    plt.tight_layout()
    out = "sa_di_4obs_from_init.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

# # your_env_setup.py
# from .double_integrator import DoubleIntegrator  # your existing GCBF+ file
# from .sa_di_wrapper import SingleAgentDIEnv
# import sys

# # 1) Construct the original env with *one* agent
# env_m = DoubleIntegrator(num_agents=1, area_size=6.0)  # other args as in your code

# # 2) Wrap it for single-agent, black-box usage
# env = SingleAgentDIEnv(env_m)


# print("car_radius:", env.car_radius)
# print("comm_radius:", env.comm_radius)
# print("n_rays:", env.n_rays)
# print("bounds:", env.action_low, env.action_high)
# # 3) Interact like a normal simulator
# obs = env.reset(seed=0)                # {'state':[x,y,vx,vy], 'goal':[gx,gy,0,0], 'state_goal':[...]}
# a0 = [0.1, -0.1]                       # acceleration inputs (fx, fy) for DI
# obs, reward, done, info = env.step(a0)


# # feed-ready features for your transformer/MLP
# x = env.feature_vector(obs)  # shape = 6 (+4 bounds) +2 radii = 12
# print("feature vector:", x.shape, x)

# print("env", env)

# print("obs, reward, done, info",obs, reward, done, info)
# sys.exit()
# # 4) MPC-style shooting / predictive check
# import numpy as np
# H = 8
# cand = np.tile(a0, (H,1))

# states, rewards, infos = env.rollout_sequence(cand, early_stop_on_violation=True)

# # 5) Utilities
# s = env.get_state()
# safe = env.is_geom_safe()              # True/False
# low, high = env.action_low, env.action_high