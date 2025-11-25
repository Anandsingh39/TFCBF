# ---- CPU guard (prevents accidental CUDA init) ----
import os as _os
if _os.getenv("PACT_DEVICE", "cpu").lower() != "cuda":
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

# Make optimizer stay on CPU path if user did not explicitly request CUDA
if _os.getenv("PACT_DEVICE", "cpu").lower() != "cuda":
    try:
        torch.cuda.is_available = lambda: False
    except Exception:
        pass

# ---- project imports (robust paths) ----
try:
    from src.models.modules.a import PACTPolicyCritic, split_state_action_embeddings, CFG  # backbone+heads
except Exception:
    from a import PACTPolicyCritic, CFG

try:
    from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv                                  # black-box wrapper
    from gcbfplus.env.double_integrator import DoubleIntegrator                              # base env

except Exception:
    try:
        from src.gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv
        from src.gcbfplus.env.double_integrator import DoubleIntegrator
    except Exception:
        from sa_di_wrapper import SingleAgentDIEnv
        from double_integrator import DoubleIntegrator


# -----------------------------
# Config
# -----------------------------
@dataclass
class OnlineConfig:
    # PACT / model
    alpha: float = 1.0
    ctx_tokens: int = 16
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 8
    device: str = "cpu"

    # Environment
    area_size: float = 4.0
    n_obs: int = 16
    max_step: int = 256
    dt: float = 0.03
    seed: int = 0

    # Safety rectifier / QP margin
    margin: float = 0.15
    margin_boost: float = 0.3
    grad_eps: float = 1e-8
    action_clip: float = 2.0

    # MPC-like verification
    verify_horizon: int = 5
    max_verify_attempts: int = 2
    
    # MPC Planner
    use_mpc: bool = False
    mpc_horizon: int = 10
    mpc_sequences: int = 16
    mpc_solver: str = "random"  # "random"
    mpc_qp_project: bool = True
    use_simulator_preview: bool = True # If True, use env.clone(); else use diff_di_step
    mpc_noise_std: float = 0.5
    w_mpc_cbf: float = 1.0
    w_mpc_smooth: float = 1e-3
    w_mpc_reward: float = 0.0
    cbf_slack: float = 0.0
    w_mpc_logsum: float = 0.1
    mpc_logsum_tau: float = 10.0

    # Alpha mixing (QP teacher vs policy)
    alpha_init: float = 0.05
    alpha_final: float = 0.90
    alpha_warmup: int = 20000
    teacher_anchor: str = "nominal"   # {"nominal", "policy"}

    # Training / optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Loss weights
    w_bce: float = 1.0
    w_margin: float = 0.5
    w_grad: float = 1e-4
    w_pi_safe: float = 0.05
    w_exec: float = 0.10     # consistency with executed mix
    w_hdot: float = 0.10     # finite-difference derivative penalty (safe states)
    w_horizon_cbf: float = 0.10 # Horizon CBF hinge loss

    # Discrete-time CBF slope (for hdot + lambda*h >= 0)
    hdot_lambda: float = 1.0

    # Online update schedule
    update_every: int = 1
    batch_size: int = 64
    buffer_capacity: int = 20000

    # Debug
    verbose: bool = True

# -----------------------------
# Buffers
# -----------------------------
class ReplayItem:
    __slots__ = (
        "state_seq", "action_seq",            # window at time t
        "state_seq_next", "action_seq_next",  # window at time t+1 (after env step)
        "u_nom", "u_pi", "u_qp", "u_exec", "u_exec_next", "alpha",
        "label_safe",
        "h_t", "h_tp1",
        "mpc_logsum", "mpc_max_violation"
    )
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class RingBuffer:
    def __init__(self, capacity:int):
        self.capacity = int(capacity)
        self.data: List[ReplayItem] = []
        self.ptr = 0
    def __len__(self):
        return len(self.data)
    def push(self, item: ReplayItem):
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.ptr] = item
            self.ptr = (self.ptr + 1) % self.capacity
    def sample(self, n:int) -> List[ReplayItem]:
        n = min(n, len(self.data))
        if n == 0: return []
        idx = np.random.choice(len(self.data), size=n, replace=False)
        return [self.data[i] for i in idx]


# -----------------------------
# Rolling context
# -----------------------------
class RollingContext:
    def __init__(self, T:int, S:int, A:int):
        self.T, self.S, self.A = T, S, A
        self._states = None
        self._actions = None
    def init(self, s0: np.ndarray, a0: Optional[np.ndarray]=None):
        s0 = np.asarray(s0, dtype=np.float32)
        if a0 is None:
            a0 = np.zeros((self.A,), dtype=np.float32)
        a0 = np.asarray(a0, dtype=np.float32)
        self._states  = np.repeat(s0[None,:], self.T, axis=0)
        self._actions = np.repeat(a0[None,:], self.T, axis=0)
    def as_batch(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.as_tensor(self._states,  dtype=torch.float32, device=device).unsqueeze(0)
        a = torch.as_tensor(self._actions, dtype=torch.float32, device=device).unsqueeze(0)
        return s, a
    def after_step_versions(self, s_next: np.ndarray, a_exec: np.ndarray):
        # build the T-length windows that would exist after applying (s_next, a_exec)
        ns = np.concatenate([self._states[1:],  s_next[None,:]], axis=0)
        na = np.concatenate([self._actions[1:], a_exec[None,:]], axis=0)
        return ns, na
    def update(self, s_next: np.ndarray, a_exec: np.ndarray):
        self._states, self._actions = self.after_step_versions(s_next, a_exec)


# -----------------------------
# Token/critic helpers
# -----------------------------
def h_score(model: PACTPolicyCritic,
            state_seq: torch.Tensor, action_seq: torch.Tensor, a_last: torch.Tensor) -> torch.Tensor:
    """
    Evaluate barrier logit hψ(s_t, a_last) by appending a_last to the action window.
    Shapes:
        state_seq:  (B,T,S)
        action_seq: (B,T,A)
        a_last:     (B,A)
    """
    B, T, A = action_seq.shape
    a_last = a_last.view(B,1,A)
    action_plus = torch.cat([action_seq[:,1:,:], a_last], dim=1)  # shift left + append
    out_plus, _ = model.pact({"state": state_seq, "action": action_plus})
    action_ctx_plus = out_plus[:,1::2,:]
    return model.critic(action_ctx_plus[:,-1,:]).squeeze(-1)  # (B,)

# check gcbf+ qp and try to add it here by tweaking if possible
def qp_teacher_anchor_nominal(model: PACTPolicyCritic,
                              state_seq: torch.Tensor,
                              action_seq: torch.Tensor,
                              u_nom: torch.Tensor,
                              margin: float,
                              eps: float=1e-8) -> torch.Tensor:
    """
    Closed-form solution of:
      min_u 0.5||u - u_nom||^2  s.t.  h(u0) + g^T (u - u0) >= margin,
    with linearization at u0 = u_nom.
    Returns (B,A) detached tensor.
    """
    u0 = u_nom.detach().clone().requires_grad_(True)  # (B,A)
    # h(u0) per-sample, and its action gradient g
    h_per = h_score(model, state_seq, action_seq, u0)      # (B,)
    h_sum = h_per.sum()
    g = torch.autograd.grad(h_sum, u0, retain_graph=False, create_graph=False)[0]  # (B,A)

    # Half-space: g^T u >= b, where b = margin - h(u0) + g^T u0
    b = (margin - h_per).unsqueeze(-1) + (g * u0).sum(-1, keepdim=True)
    denom = (g * g).sum(-1, keepdim=True).clamp_min(eps)
    # Project u_nom onto the half-space
    gt_unom = (g * u0).sum(-1, keepdim=True)
    lam = ((b - gt_unom) / denom).clamp_min(0.0)
    u_qp = u0 + lam * g
    return u_qp.detach()


def diff_di_step(state: torch.Tensor, action: torch.Tensor, dt: float, m: float=0.1) -> torch.Tensor:
    """
    Differentiable Double Integrator dynamics for training loss.
    state: (B, S) [x, y, vx, vy, dx, dy, lidar...]
    action: (B, 2) [fx, fy]
    """
    x, y, vx, vy = state[:,0], state[:,1], state[:,2], state[:,3]
    dx, dy = state[:,4], state[:,5]
    lidar = state[:,6:]

    ax = action[:,0] / m
    ay = action[:,1] / m

    # Euler integration
    x_next = x + vx * dt
    y_next = y + vy * dt
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt

    # Goal relative
    dx_next = dx - vx * dt
    dy_next = dy - vy * dt

    # Reassemble
    next_state = torch.cat([
        x_next[:,None], y_next[:,None],
        vx_next[:,None], vy_next[:,None],
        dx_next[:,None], dy_next[:,None],
        lidar # Lidar assumed static for gradient approximation
    ], dim=1)
    return next_state


# -----------------------------
# Online runner
# -----------------------------
class OnlineRunner:
    def __init__(self, cfg: OnlineConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Env
        params = DoubleIntegrator.PARAMS.copy()
        params["n_obs"] = cfg.n_obs
        base = DoubleIntegrator(num_agents=1, area_size=cfg.area_size, max_step=cfg.max_step, dt=cfg.dt, params=params)
        self.env = SingleAgentDIEnv(base, include_velocity=True, normalize_lidar=True, full_obs_keys=("state_goal","lidar"))
        obs = self.env.reset(seed=cfg.seed)
        full_obs = np.asarray(obs["full_obs"], dtype=np.float32)
        # print(full_obs)
        S = int(full_obs.shape[0])
        A = 2

        # Model
        self.model = PACTPolicyCritic(state_dim=S, action_dim=A, ctx_tokens=cfg.ctx_tokens,
                                      n_embd=cfg.n_embd, n_layer=cfg.n_layer, n_head=cfg.n_head,
                                      action_input_type="continuous").to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Buffers
        cap = cfg.buffer_capacity
        self.primary = RingBuffer(cap); self.safe_buf = RingBuffer(cap); self.unsafe_buf = RingBuffer(cap)

        # Rolling context
        T = cfg.ctx_tokens // 2
        self.ctx = RollingContext(T=T, S=S, A=A)
        self.ctx.init(s0=full_obs, a0=np.zeros((A,), dtype=np.float32))

        # Bounds
        self.act_low  = obs.get("action_low",  np.array([-cfg.action_clip, -cfg.action_clip], dtype=np.float32))
        self.act_high = obs.get("action_high", np.array([ cfg.action_clip,  cfg.action_clip], dtype=np.float32))

        self.global_step = 0
        self.episode_count = 0

    # ---- helpers ----
    def alpha(self) -> float:
        if self.cfg.alpha_warmup <= 0: return float(self.cfg.alpha_final)
        r = min(1.0, self.global_step / float(self.cfg.alpha_warmup))
        return float(self.cfg.alpha_init + r * (self.cfg.alpha_final - self.cfg.alpha_init))

    def nominal_action(self) -> np.ndarray:
        if hasattr(self.env, "nominal_action"):
            u = self.env.nominal_action()
            return np.asarray(u, dtype=np.float32).reshape(-1).copy()
        return np.zeros((2,), dtype=np.float32)

    def verify_horizon(self, a0: np.ndarray, H: int) -> bool:
        if not hasattr(self.env, "rollout_sequence"):
            # Fallback: use clone if available
            if hasattr(self.env, "clone"):
                sim = self.env.clone()
                cand = np.tile(a0.reshape(1,-1), (H,1)).astype(np.float32)
                for k in range(H):
                    _, _, _, info = sim.step(cand[k])
                    if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
                        return False
                return True
            return True
        cand = np.tile(a0.reshape(1,-1), (H,1)).astype(np.float32)
        states, rewards, infos = self.env.rollout_sequence(cand, early_stop_on_violation=False)
        for k in range(len(infos)):
            inf = infos[k] if isinstance(infos, (list,tuple)) else infos
            if isinstance(inf, dict) and (inf.get("inside_obstacles") or inf.get("unsafe") or inf.get("collision")):
                return False
        return True

    def _cbf_residuals_discrete(self, h_seq: torch.Tensor) -> torch.Tensor:
        dt = max(self.cfg.dt, 1e-6)
        lam = self.cfg.hdot_lambda
        slack = getattr(self.cfg, "cbf_slack", 0.0)
        return (h_seq[1:] - h_seq[:-1]) / dt + lam * h_seq[:-1] - slack

    def _plan_cost(self, state_seq: torch.Tensor, action_seq: torch.Tensor, U_np: np.ndarray) -> float:
        cfg = self.cfg
        dev = self.device
        if U_np.ndim != 2:
            U_np = U_np.reshape(-1, action_seq.shape[-1])
        U_np = np.clip(U_np, self.act_low[None,:], self.act_high[None,:]).astype(np.float32)
        H = U_np.shape[0]

        use_sim = cfg.use_simulator_preview and hasattr(self.env, "clone")
        sim = None
        if use_sim:
            try:
                sim = self.env.clone()
            except Exception:
                sim = None

        states_win = state_seq[0].detach().cpu().numpy().copy()
        actions_win = action_seq[0].detach().cpu().numpy().copy()

        rewards: List[float] = []
        unsafe_penalty = 0.0

        with torch.no_grad():
            h_vals: List[torch.Tensor] = []

            for k in range(H):
                curr_action = U_np[k]
                actions_win = np.concatenate([actions_win[1:], curr_action[None,:]], axis=0)

                st_t = torch.as_tensor(states_win, dtype=torch.float32, device=dev).unsqueeze(0)
                ac_t = torch.as_tensor(actions_win, dtype=torch.float32, device=dev).unsqueeze(0)
                a_t = torch.as_tensor(curr_action[None,:], dtype=torch.float32, device=dev)
                h_curr = h_score(self.model, st_t, ac_t, a_t).squeeze(0)
                h_vals.append(h_curr)

                next_state_tensor: Optional[torch.Tensor] = None
                if sim is not None:
                    try:
                        obs, rew, done, info = sim.step(curr_action)
                        rewards.append(float(rew))
                        info = info or {}
                        if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
                            unsafe_penalty = max(unsafe_penalty, 1.0)
                        next_full = obs.get("full_obs") if isinstance(obs, dict) else None
                        if next_full is None and isinstance(obs, dict):
                            next_full = obs.get("state")
                        if next_full is None:
                            next_full = states_win[-1]
                        next_state_tensor = torch.as_tensor(next_full, dtype=torch.float32, device=dev)
                    except Exception:
                        sim = None
                if next_state_tensor is None:
                    last_state = torch.as_tensor(states_win[-1], dtype=torch.float32, device=dev).unsqueeze(0)
                    action_tensor = torch.as_tensor(curr_action[None,:], dtype=torch.float32, device=dev)
                    next_state = diff_di_step(last_state, action_tensor, cfg.dt).squeeze(0)
                else:
                    next_state = next_state_tensor

                states_win = np.concatenate([states_win[1:], next_state.detach().cpu().numpy()[None,:]], axis=0)

            last_action = U_np[-1]
            actions_win_final = np.concatenate([actions_win[1:], last_action[None,:]], axis=0)
            st_final = torch.as_tensor(states_win, dtype=torch.float32, device=dev).unsqueeze(0)
            ac_final = torch.as_tensor(actions_win_final, dtype=torch.float32, device=dev).unsqueeze(0)
            a_final = torch.as_tensor(last_action[None,:], dtype=torch.float32, device=dev)
            h_final = h_score(self.model, st_final, ac_final, a_final).squeeze(0)
            h_vals.append(h_final)

            h_seq = torch.stack(h_vals, dim=0)
            residuals = self._cbf_residuals_discrete(h_seq)
            cbf_hinge = torch.relu(-residuals).mean()
            cbf_cost = float(cbf_hinge.detach().cpu().item())

        smooth = float(np.sum(np.square(np.diff(U_np, axis=0)))) if H > 1 else 0.0
        reward_term = -float(np.sum(rewards)) if rewards else 0.0
        penalty = 1000.0 * unsafe_penalty

        violations = torch.relu(-residuals)
        max_violation = float(violations.max().detach().cpu().item()) if violations.numel() > 0 else 0.0
        tau = max(float(getattr(cfg, "mpc_logsum_tau", 10.0)), 1e-6)
        logsum_violation = float((torch.logsumexp(tau * violations, dim=0) / tau).detach().cpu().item()) if violations.numel() > 0 else 0.0

        total = cfg.w_mpc_cbf * cbf_cost
        total += cfg.w_mpc_smooth * smooth
        total += cfg.w_mpc_reward * reward_term
        total += penalty

        return {
            "total": total,
            "cbf_cost": cbf_cost,
            "max_violation": max_violation,
            "logsum_violation": logsum_violation,
            "smooth": smooth,
            "reward_term": reward_term,
            "penalty": penalty
        }

    def mpc_planner(self, state_seq: torch.Tensor, action_seq: torch.Tensor, u_nom: torch.Tensor, u_pi: torch.Tensor) -> torch.Tensor:
        """
        Model-free MPC planner using simulator preview (env.clone()).
        Evaluates:
        1. Policy Constant (open-loop)
        2. Nominal Constant (open-loop)
        3. Nominal Closed-Loop (re-query nominal at each step)
        4. Random perturbations around Policy (optional)
        """
        cfg = self.cfg
        B, T, S = state_seq.shape
        _, _, A = action_seq.shape
        H = cfg.mpc_horizon
        N = cfg.mpc_sequences

        u_pi_np = u_pi.detach().cpu().numpy().reshape(-1)
        u_nom_np = u_nom.detach().cpu().numpy().reshape(-1)

        base_policy = np.tile(u_pi_np[None,:], (H,1)).astype(np.float32)
        base_nominal = np.tile(u_nom_np[None,:], (H,1)).astype(np.float32)

        best_cost = float("inf")
        best_seq = base_policy
        best_metrics = None

        for i in range(N):
            if i == 0:
                U = base_policy.copy()
            elif i == 1:
                U = base_nominal.copy()
            else:
                noise = np.random.normal(0.0, cfg.mpc_noise_std, size=(H, A)).astype(np.float32)
                U = base_policy + noise

            metrics = self._plan_cost(state_seq, action_seq, U)
            cost = float(metrics["total"])
            if cost < best_cost:
                best_cost = cost
                best_seq = U
                best_metrics = metrics

        if best_metrics is None:
            best_metrics = {
                "total": best_cost,
                "cbf_cost": 0.0,
                "max_violation": 0.0,
                "logsum_violation": 0.0,
                "smooth": 0.0,
                "reward_term": 0.0,
                "penalty": 0.0,
            }

        return {
            "best_sequence": best_seq,
            "best_cost": best_cost,
            "metrics": best_metrics
        }

    # ---- one control step ----
    def step_once(self, t:int) -> Dict[str, Any]:
        cfg = self.cfg; dev = self.device

        # Windows
        state_seq, action_seq = self.ctx.as_batch(dev)  # (1,T,S), (1,T,A)

        # Nominal LQR
        u_nom_np = self.nominal_action()
        u_nom = torch.as_tensor(u_nom_np, device=dev).unsqueeze(0)  # (1,2)
        # Policy proposal
        with torch.no_grad():
            out, _ = self.model.pact({"state": state_seq, "action": action_seq})

            last_state = out[:,0::2,:][:,-1,:]  
            delta = self.model.policy(last_state)  # (1,2)
            u_pi = torch.clamp(u_nom + delta,
                               min=torch.as_tensor(self.act_low,  device=dev),
                               max=torch.as_tensor(self.act_high, device=dev))

        mpc_eval = None
        if cfg.use_mpc:
            mpc_eval = self.mpc_planner(state_seq, action_seq, u_nom, u_pi)

        # QP teacher anchored at nominal
        u_qp = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom, margin=cfg.margin, eps=cfg.grad_eps)

        # Alpha-mixed action (policy + rectified teacher)
        alpha = self.alpha()
        u_exec = (1.0 - alpha)*u_qp + alpha*u_pi
        
        u_exec = torch.clamp(u_exec,
                             min=torch.as_tensor(self.act_low,  device=dev),
                             max=torch.as_tensor(self.act_high, device=dev))

        # Horizon verification & repairs
        a_try = u_exec.squeeze(0).cpu().numpy()
        ok = self.verify_horizon(a_try, cfg.verify_horizon)
        if not ok and cfg.max_verify_attempts > 0:
            # stronger margin
            u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom, margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps)
            u_exec2 = ((1.0 - alpha)*u_qp2 + alpha*u_pi).clamp(
                torch.as_tensor(self.act_low, device=dev),
                torch.as_tensor(self.act_high, device=dev)
            )
            a_try2 = u_exec2.squeeze(0).cpu().numpy()
            if self.verify_horizon(a_try2, cfg.verify_horizon):
                u_exec = u_exec2
            else:
                # fall back to pure QP if that passes
                if self.verify_horizon(u_qp.squeeze(0).cpu().numpy(), cfg.verify_horizon):
                    u_exec = u_qp
                else:
                    # last resort: nominal if preview says ok
                    if self.verify_horizon(u_nom.squeeze(0).cpu().numpy(), cfg.verify_horizon):
                        u_exec = u_nom

        # Critic reads
        with torch.no_grad():
            h_pi  = h_score(self.model, state_seq, action_seq, u_pi).item()
            h_qp  = h_score(self.model, state_seq, action_seq, u_qp).item()
            h_exe = h_score(self.model, state_seq, action_seq, u_exec).item()

        mpc_logsum = 0.0
        mpc_max = 0.0
        if mpc_eval is not None:
            metrics = mpc_eval.get("metrics", {}) if isinstance(mpc_eval, dict) else {}
            mpc_logsum = float(metrics.get("logsum_violation", 0.0))
            mpc_max = float(metrics.get("max_violation", 0.0))

        # Execute in real env
        a_exec_np = u_exec.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = self.env.step(a_exec_np)
        label_safe = 1
        if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
            label_safe = 0
        if done and label_safe == 1:
            # treat terminal as unsafe unless explicitly marked safe
            label_safe = 0

        # Next-window versions (for hdot)
        s_next = np.asarray(next_obs["full_obs"], dtype=np.float32)
        state_seq_next_np, action_seq_next_np = self.ctx.after_step_versions(s_next, a_exec_np)
        state_seq_next = torch.as_tensor(state_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)
        action_seq_next= torch.as_tensor(action_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)

        with torch.no_grad():
            h_tp1 = h_score(self.model, state_seq_next, action_seq_next, u_exec).item()

        # Update context
        self.ctx.update(s_next, a_exec_np)

        self.global_step += 1

        return {
            "t": t, "reward": float(reward), "done": bool(done),
            "state_seq": state_seq.squeeze(0).cpu().numpy(),
            "action_seq": action_seq.squeeze(0).cpu().numpy(),
            "state_seq_next": state_seq_next.squeeze(0).cpu().numpy(),
            "action_seq_next": action_seq_next.squeeze(0).cpu().numpy(),
            "u_nom": u_nom.squeeze(0).cpu().numpy(),
            "u_pi": u_pi.squeeze(0).cpu().numpy(),
            "u_qp": u_qp.squeeze(0).cpu().numpy(),
            "u_exec": a_exec_np.astype(np.float32),
            "alpha": float(alpha),
            "h_t": float(h_exe),
            "h_tp1": float(h_tp1),
            "info": info,
            "mpc_logsum": mpc_logsum,
            "mpc_max_violation": mpc_max,
        }

    # ---- sampling / training ----
    def sample_minibatch(self, batch_size:int) -> Dict[str, torch.Tensor]:
        n_unsafe = min(len(self.unsafe_buf), batch_size//2)
        n_safe   = batch_size - n_unsafe
        items: List[ReplayItem] = []
        if n_safe>0 and len(self.safe_buf)>0:   items += self.safe_buf.sample(n_safe)
        if n_unsafe>0 and len(self.unsafe_buf)>0: items += self.unsafe_buf.sample(n_unsafe)
        if len(items)==0: items = self.primary.sample(batch_size)
        if len(items)==0: return {}

        dev = self.device

        def stack(attr):
            vals = [getattr(it, attr) for it in items]
            if isinstance(vals[0], np.ndarray):
                arr = np.stack(vals, axis=0)
            else:
                arr = np.array(vals, dtype=np.float32)
            return torch.as_tensor(arr, dtype=torch.float32, device=dev)

        batch = {
            "state_seq":       stack("state_seq"),
            "action_seq":      stack("action_seq"),
            "state_seq_next":  stack("state_seq_next"),
            "action_seq_next": stack("action_seq_next"),
            "u_nom":           stack("u_nom"),
            "u_pi":            stack("u_pi"),
            "u_qp":            stack("u_qp"),
            "u_exec":          stack("u_exec"),
            "u_exec_next":     stack("u_exec_next"),
            "y":               torch.as_tensor(np.array([it.label_safe for it in items], dtype=np.float32), device=dev),
            "mpc_logsum":      stack("mpc_logsum"),
            "mpc_max_violation": stack("mpc_max_violation"),
        }
        return batch

    def train_step(self) -> Dict[str, float]:
        batch = self.sample_minibatch(self.cfg.batch_size)
        if not batch:
            return {"loss": 0.0}

        st   = batch["state_seq"]
        ac   = batch["action_seq"]
        stn  = batch["state_seq_next"]
        acn  = batch["action_seq_next"]
        u_nom= batch["u_nom"]
        u_pi = batch["u_pi"]
        u_qp = batch["u_qp"]
        u_ex = batch["u_exec"]
        u_ex_next = batch["u_exec_next"]
        y    = batch["y"]
        dev  = self.device

        # Forward backbone
        out, _ = self.model.pact({"state": st, "action": ac})
        last_state = out[:,0::2,:][:,-1,:]

        # Policy current prediction
        delta = self.model.policy(last_state)
        u_pred = torch.clamp(u_nom + delta,
                             min=torch.as_tensor(self.act_low,  device=dev),
                             max=torch.as_tensor(self.act_high, device=dev))

        # Critic at u_exec (for barrier training)
        h_k = h_score(self.model, st, ac, u_ex)
        
        # Critic at u_exec_next (for h_dot)
        h_kp1 = h_score(self.model, stn, acn, u_ex_next)

        # Loss components
        gamma = self.cfg.margin
        alpha_cbf = self.cfg.alpha

        # 1. Safe/Unsafe classification
        # Safe (y=1): h > 0 -> relu(gamma - h)
        # Unsafe (y=0): h < 0 -> relu(gamma + h)
        L_safe_cls = y * torch.relu(gamma - h_k)
        L_unsafe_cls = (1.0 - y) * torch.relu(gamma + h_k)
        L_cls = (L_safe_cls + L_unsafe_cls).mean()

        # 2. CBF condition: h_dot + alpha * h >= 0
        # h_dot = (h(k+1) - h(k)) / dt
        # relu(gamma - (hdot + alpha * h))
        hdot = (h_kp1 - h_k) / max(self.cfg.dt, 1e-6)
        L_hdot = torch.relu(gamma - (hdot + alpha_cbf * h_k)).mean()

        # 3. Horizon CBF hinge loss (Critic-only rollout)
        # Rollout u_pred for H steps using diff_di_step
        # We use u_pred (policy action) as constant action for the horizon?
        # Or we can just do 1 step lookahead if H=1?
        # The user asked for "horizon CBF hinge penalties".
        # We will rollout for mpc_horizon steps.
        L_horizon = torch.tensor(0.0, device=dev)
        L_mpc_future = torch.tensor(0.0, device=dev)
        if self.cfg.w_horizon_cbf > 0 or self.cfg.w_mpc_logsum > 0:
            H_train = self.cfg.mpc_horizon
            sim_st = st
            sim_ac = ac
            h_curr = h_k # h(s_t, a_{t-1})? No, h_k is h(s_t, u_ex).
            # We want to evaluate policy action u_pred.
            # So we need h(s_t, u_pred).
            h_curr_pi = h_score(self.model, st, ac, u_pred)
            
            curr_u = u_pred
            viol_list: List[torch.Tensor] = []
            
            for k in range(H_train):
                # Rollout state
                last_s = sim_st[:, -1, :]
                s_next = diff_di_step(last_s, curr_u, self.cfg.dt)
                
                # Update history
                sim_st = torch.cat([sim_st[:, 1:, :], s_next.unsqueeze(1)], dim=1)
                sim_ac = torch.cat([sim_ac[:, 1:, :], curr_u.unsqueeze(1)], dim=1)
                
                # Evaluate h
                out_next, _ = self.model.pact({"state": sim_st, "action": sim_ac})
                h_next = self.model.critic(out_next[:,1::2,:][:,-1,:]).squeeze(-1)
                
                # Loss
                h_dot_pi = (h_next - h_curr_pi) / self.cfg.dt
                viol = torch.relu(gamma - (h_dot_pi + alpha_cbf * h_curr_pi))
                viol_list.append(viol)
                
                h_curr_pi = h_next
                # Assume constant action u_pred for the horizon
                # (Autoregressive policy rollout is too expensive here)

            if viol_list:
                viol_stack = torch.stack(viol_list, dim=0)  # (H,B)
                if self.cfg.w_horizon_cbf > 0:
                    L_horizon = viol_stack.mean()
                if self.cfg.w_mpc_logsum > 0:
                    tau = max(float(self.cfg.mpc_logsum_tau), 1e-6)
                    scaled = viol_stack / tau
                    L_mpc_future = (torch.logsumexp(scaled, dim=0) * tau).mean()

        # grad penalty for smoother rectifier
        u_pred_req = u_pred.detach().requires_grad_(True)
        h_tmp = h_score(self.model, st, ac, u_pred_req)
        g = torch.autograd.grad(h_tmp.sum(), u_pred_req, retain_graph=False, create_graph=False)[0]
        L_grad = (g*g).sum(dim=-1).mean()

        # QP teacher on the batch (anchor nominal)
        u_qp_batch = qp_teacher_anchor_nominal(self.model, st, ac, u_nom, margin=self.cfg.margin, eps=self.cfg.grad_eps)

        # Alpha (mixing parameter)
        alpha_mix = torch.tensor(self.alpha(), device=dev)

        # Policy losses
        L_teacher = ((u_pred - u_qp_batch)**2).sum(dim=-1) * (1.0 - y)
        L_safe    = ((u_pred - u_nom    )**2).sum(dim=-1) * y
        u_mix_det = ((1.0 - alpha_mix)*u_qp_batch + alpha_mix*u_pred.detach())
        L_exec    = ((u_pred - u_mix_det)**2).sum(dim=-1)
        L_pi = (1.0 - alpha_mix) * L_teacher.mean() + self.cfg.w_pi_safe * L_safe.mean() + self.cfg.w_exec * L_exec.mean()

        loss = self.cfg.w_margin * L_cls + self.cfg.w_hdot * L_hdot + self.cfg.w_grad * L_grad + L_pi
        loss = loss + self.cfg.w_horizon_cbf * L_horizon + self.cfg.w_mpc_logsum * L_mpc_future

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "L_cls": float(L_cls.item()),
            "L_hdot": float(L_hdot.item()),
            "L_grad": float(L_grad.item()),
            "L_pi": float(L_pi.item()),
            "L_hor": float(L_horizon.item()),
            "L_mpc": float(L_mpc_future.item()),
        }

    def run_episode(self, max_steps: Optional[int]=None) -> Dict[str, Any]:
        max_steps = max_steps or self.cfg.max_step
        # Increment seed each episode for diversity
        episode_seed = self.cfg.seed + self.episode_count * 10007  # Large prime for diversity
        obs = self.env.reset(seed=episode_seed)
        self.ctx.init(obs["full_obs"].astype(np.float32), np.zeros((2,), dtype=np.float32))

        state_arr = np.asarray(obs.get("state", obs["full_obs"][:2]), dtype=np.float32)
        start_pos = state_arr[:2].copy()

        ep_reward = 0.0
        ep_cost = 0.0
        logs = []
        episode_data = []
        
        final_reward = 0.0
        is_unsafe = False
        finished = False

        # Check initial state for finish
        if hasattr(self.env, "finish_mask"):
            if self.env.finish_mask():
                finished = True

        for t in range(max_steps):
            step_out = self.step_once(t)
            ep_reward += float(step_out["reward"])
            final_reward = float(step_out["reward"])
            
            # Metrics from info
            info = step_out["info"]
            cost = info.get("cost", 0.0)
            ep_cost += cost
            if cost > 1e-6:
                is_unsafe = True
            
            # Check finish
            if hasattr(self.env, "finish_mask"):
                if self.env.finish_mask():
                    finished = True
            
            episode_data.append(step_out)

            train_out = {"loss": 0.0}
            if self.cfg.update_every > 0 and (t % self.cfg.update_every == 0):
                train_out = self.train_step()

            logs.append({**step_out, **train_out})
            if self.cfg.verbose and (t % 20 == 0):
                print(f"[t={t:03d}] r={step_out['reward']:.3f} "
                      f"h_exec={step_out['h_t']:.3f} "
                      f"alpha={step_out['alpha']:.2f} "
                      f"loss={train_out.get('loss',0.0):.4f}")
            if step_out["done"]:
                break

        # Labeling
        H = self.cfg.verify_horizon
        T_ep = len(episode_data)
        
        unsafe_mask = np.zeros(T_ep, dtype=int)
        for i, data in enumerate(episode_data):
            info = data["info"]
            if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
                unsafe_mask[i] = 1
        
        safe_mask = np.ones(T_ep, dtype=int)
        for i in range(T_ep):
            end = min(i + H, T_ep)
            if np.any(unsafe_mask[i:end]):
                safe_mask[i] = 0
        
        for i, data in enumerate(episode_data):
            if i < T_ep - 1:
                u_exec_next = episode_data[i+1]["u_exec"]
            else:
                u_exec_next = np.zeros_like(data["u_exec"])
            
            item = ReplayItem(
                state_seq=data["state_seq"],
                action_seq=data["action_seq"],
                state_seq_next=data["state_seq_next"],
                action_seq_next=data["action_seq_next"],
                u_nom=data["u_nom"],
                u_pi=data["u_pi"],
                u_qp=data["u_qp"],
                u_exec=data["u_exec"],
                u_exec_next=u_exec_next,
                alpha=data["alpha"],
                label_safe=int(safe_mask[i]),
                h_t=data["h_t"],
                h_tp1=data["h_tp1"],
                mpc_logsum=data.get("mpc_logsum", 0.0),
                mpc_max_violation=data.get("mpc_max_violation", 0.0)
            )
            self.primary.push(item)
            (self.safe_buf if safe_mask[i]==1 else self.unsafe_buf).push(item)

        # Extract positions for plotting
        goal_pos = np.asarray(obs["goal"][:2], dtype=np.float32)
        if len(episode_data) > 0:
            final_pos = np.asarray(episode_data[-1]["state_seq_next"][-1, :2], dtype=np.float32)
        else:
            final_pos = np.asarray(obs.get("state", obs["full_obs"][:2]), dtype=np.float32)[:2]

        # Increment episode counter for next episode
        self.episode_count += 1

        metrics = {
            "ep_reward": ep_reward,
            "steps": len(episode_data),
            "logs": logs,
            "eval/reward": ep_reward,
            "eval/reward_final": final_reward,
            "eval/cost": ep_cost,
            "eval/unsafe_frac": 1.0 if is_unsafe else 0.0,
            "eval/finish": 1.0 if finished else 0.0,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "final_pos": final_pos
        }
        return metrics


# -----------------------------
# Convenience API
# -----------------------------
def run_online_episode(cfg: Optional[OnlineConfig]=None) -> Dict[str, Any]:
    cfg = cfg or OnlineConfig()
    _os.environ["PACT_DEVICE"] = cfg.device
    runner = OnlineRunner(cfg)
    return runner.run_episode()


# # =============================
# # Main Entry Point
# # =============================
def main():
    """
    Main function to run the online training loop.
    
    Process:
    1. Parse command-line arguments (if any)
    2. Create configuration with default or user-specified parameters
    3. Initialize the OnlineRunner (which sets up env, model, buffers)
    4. Run training episodes in a loop
    5. Log progress after each episode
    
    The agent learns continuously as it interacts with the environment,
    with no separate data collection or offline training phase.
    """
    # Create configuration with default hyperparameters
    cfg = OnlineConfig()
  #  print("Config:", cfg)

    # Initialize the training runner
    # This creates the environment, model, optimizer, and buffers
    runner = OnlineRunner(cfg)

    # === Main Training Loop ===
    # Run multiple episodes to train the agent
    for episode in range(5000):
        # Run one complete episode (agent acts until done or max_step reached)
        # The runner handles:
        # - Resetting the environment
        # - Getting actions from policy/teacher
        # - Executing actions in the environment
        # - Storing experiences in replay buffer
        # - Triggering training updates
        ep_out = runner.run_episode()
        
        # Log episode statistics
        s_fmt = f"({ep_out['start_pos'][0]:.2f}, {ep_out['start_pos'][1]:.2f})"
        g_fmt = f"({ep_out['goal_pos'][0]:.2f}, {ep_out['goal_pos'][1]:.2f})"
        f_fmt = f"({ep_out['final_pos'][0]:.2f}, {ep_out['final_pos'][1]:.2f})"
        print(f"Ep {episode}: Steps={ep_out['steps']}, Reward={ep_out['ep_reward']:.2f}, Start={s_fmt}, Goal={g_fmt}, Final={f_fmt}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--H", type=int, default=5)
    p.add_argument("--alpha-init", type=float, default=0.05)
    p.add_argument("--alpha-final", type=float, default=0.9)
    p.add_argument("--alpha-warmup", type=int, default=20000)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--n_obs", type=int, default=8)
    p.add_argument("--area_size", type=float, default=6.0)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--use-mpc", action="store_true")
    p.add_argument("--mpc-horizon", type=int, default=10)
    p.add_argument("--mpc-sequences", type=int, default=16)
    args = p.parse_args()

    _os.environ["PACT_DEVICE"] = args.device

    cfg = OnlineConfig(
        device=args.device, max_step=args.steps, seed=args.seed,
        verify_horizon=args.H, verbose=args.verbose,
        alpha_init=args.alpha_init, alpha_final=args.alpha_final, alpha_warmup=args.alpha_warmup,
        n_obs=args.n_obs, area_size=args.area_size,
        use_mpc=args.use_mpc, mpc_horizon=args.mpc_horizon, mpc_sequences=args.mpc_sequences
    )
    
    runner = OnlineRunner(cfg)
    
    # Accumulate metrics
    total_metrics = {
        "eval/reward": [],
        "eval/reward_final": [],
        "eval/cost": [],
        "eval/unsafe_frac": [],
        "eval/finish": []
    }
    
    starts = []
    goals = []
    finals = []

    for i in range(args.num_episodes):
        print(f"--- Episode {i+1}/{args.num_episodes} ---")
        out = runner.run_episode()
        print(f"Episode {i+1} reward={out['ep_reward']:.3f} steps={out['steps']}")
        print(f"  Metrics: Reward={out['eval/reward']:.3f}, Cost={out['eval/cost']:.3f}, "
              f"Unsafe={out['eval/unsafe_frac']:.0f}, Finish={out['eval/finish']:.0f}")
        print(f"  Start=({out['start_pos'][0]:.2f}, {out['start_pos'][1]:.2f}) "
              f"Goal=({out['goal_pos'][0]:.2f}, {out['goal_pos'][1]:.2f}) "
              f"Final=({out['final_pos'][0]:.2f}, {out['final_pos'][1]:.2f})")
        
        for k in total_metrics:
            total_metrics[k].append(out[k])
            
        starts.append(out["start_pos"])
        goals.append(out["goal_pos"])
        finals.append(out["final_pos"])

    # Summary
    print("\n=== Evaluation Summary ===")
    for k, v in total_metrics.items():
        print(f"{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
        
    # Plotting
    try:
        starts = np.array(starts)
        goals = np.array(goals)
        finals = np.array(finals)
        plt.figure(figsize=(6, 6))
        plt.scatter(starts[:, 0], starts[:, 1], c='r', marker='x', s=60, label='Start')
        plt.scatter(goals[:, 0], goals[:, 1], c='g', marker='*', s=100, label='Goal')
        plt.scatter(finals[:, 0], finals[:, 1], c='b', marker='o', label='Final Pos')
        
        # Draw lines connecting goal and final
        for g, f in zip(goals, finals):
            plt.plot([g[0], f[0]], [g[1], f[1]], 'k--', alpha=0.3)
            
        plt.xlim(-args.area_size/2, args.area_size/2)
        plt.ylim(-args.area_size/2, args.area_size/2)
        plt.grid(True)
        plt.legend()
        plt.title("Goal vs Final Position")
        plt.savefig("goal_vs_final.png")
        print("Plot saved to goal_vs_final.png")
    except Exception as e:
        print(f"Could not plot: {e}")
# # ---- CPU guard (prevents accidental CUDA init) ----
# import os as _os
# if _os.getenv("PACT_DEVICE", "cpu").lower() != "cuda":
#     _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# import time
# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, Any, List
# import numpy as np
# import torch
# import torch.nn as nn

# # Make optimizer stay on CPU path
# if _os.getenv("PACT_DEVICE", "cpu").lower() != "cuda":
#     try:
#         torch.cuda.is_available = lambda: False
#     except Exception:
#         pass

# # ---- project imports (robust paths) ----
# try:
#     from src.models.modules.a import PACTPolicyCritic, CFG
# except Exception:
#     from a import PACTPolicyCritic, CFG

# try:
#     from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv
#     from gcbfplus.env.double_integrator import DoubleIntegrator
# except Exception:
#     try:
#         from src.gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv
#         from src.gcbfplus.env.double_integrator import DoubleIntegrator
#     except Exception:
#         from sa_di_wrapper import SingleAgentDIEnv
#         from double_integrator import DoubleIntegrator


# # -----------------------------
# # Config
# # -----------------------------
# @dataclass
# class OnlineConfig:
#     # PACT / model
#     ctx_tokens: int = 16
#     n_embd: int = 128
#     n_layer: int = 4
#     n_head: int = 8
#     device: str = "cpu"

#     # Environment
#     area_size: float = 6.0
#     max_step: int = 256
#     dt: float = 0.03
#     seed: int = 0

#     # Safety rectifier
#     margin: float = 0.15
#     grad_eps: float = 1e-8
#     action_clip: float = 2.0

#     # MPC-like verification
#     verify_horizon: int = 5          # <---- horizon H
#     max_verify_attempts: int = 2     # try stronger rectification if horizon check fails
#     margin_boost: float = 0.3        # add to margin on 2nd attempt

#     # Training
#     lr: float = 3e-4
#     weight_decay: float = 1e-4
#     grad_clip: float = 1.0

#     # Loss weights
#     w_bce: float = 1.0
#     w_margin: float = 0.5
#     w_grad: float = 1e-4
#     w_pi_fix: float = 1.0
#     w_pi_safe: float = 0.05

#     # Online update schedule
#     update_every: int = 1
#     batch_size: int = 64
#     buffer_capacity: int = 20000

#     # Debug
#     verbose: bool = True


# # -----------------------------
# # Buffers
# # -----------------------------
# class ReplayItem:
#     __slots__ = ("state_seq","action_seq","u_nom","u_pred","u_exec","label_safe",
#                  "h_pred","h_exec")
#     def __init__(self, state_seq, action_seq, u_nom, u_pred, u_exec, label_safe, h_pred, h_exec):
#         self.state_seq  = state_seq
#         self.action_seq = action_seq
#         self.u_nom      = u_nom
#         self.u_pred     = u_pred
#         self.u_exec     = u_exec
#         self.label_safe = label_safe
#         self.h_pred     = h_pred
#         self.h_exec     = h_exec

# class RingBuffer:
#     def __init__(self, capacity:int):
#         self.capacity = int(capacity)
#         self.data: List[ReplayItem] = []
#         self.ptr = 0
#     def __len__(self):
#         return len(self.data)
#     def push(self, item: ReplayItem):
#         if len(self.data) < self.capacity:
#             self.data.append(item)
#         else:
#             self.data[self.ptr] = item
#             self.ptr = (self.ptr + 1) % self.capacity
#     def sample(self, batch_size:int) -> List[ReplayItem]:
#         n = min(batch_size, len(self.data))
#         if n == 0:
#             return []
#         idx = np.random.choice(len(self.data), size=n, replace=False)
#         return [self.data[i] for i in idx]


# # -----------------------------
# # Rolling context
# # -----------------------------
# class RollingContext:
#     def __init__(self, T:int, S:int, A:int, device:str="cpu"):
#         self.T, self.S, self.A = T, S, A
#         self.device = device
#         self._states  = None  # (T,S)
#         self._actions = None  # (T,A)
#     def init(self, s0: np.ndarray, a0: Optional[np.ndarray]=None):
#         s0 = np.asarray(s0, dtype=np.float32)
#         if a0 is None:
#             a0 = np.zeros((self.A,), dtype=np.float32)
#         a0 = np.asarray(a0, dtype=np.float32)
#         self._states  = np.repeat(s0[None,:], self.T, axis=0)
#         self._actions = np.repeat(a0[None,:], self.T, axis=0)
#     def as_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         s = torch.as_tensor(self._states,  dtype=torch.float32).unsqueeze(0)
#         a = torch.as_tensor(self._actions, dtype=torch.float32).unsqueeze(0)
#         return s, a
#     def update_after_step(self, s_next: np.ndarray, a_exec: np.ndarray):
#         self._states  = np.concatenate([self._states[1:],  s_next[None,:]], axis=0)
#         self._actions = np.concatenate([self._actions[1:], a_exec[None,:]], axis=0)


# # -----------------------------
# # Small helpers (appending actions, critic calls, nominal)
# # -----------------------------
# def append_action_and_encode(pact_model: PACTPolicyCritic,
#                              state_seq: torch.Tensor,   # (B,T,S)
#                              action_seq: torch.Tensor,  # (B,T,A)
#                              a_last: torch.Tensor       # (B,A)
#                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Returns:
#       last_action_ctx: (B,d)
#       state_ctx_plus, action_ctx_plus: (B,T,d)
#     """
#     B, T, A = action_seq.shape
#     a_last = a_last.view(B,1,A)
#     action_plus = torch.cat([action_seq[:,1:,:], a_last], dim=1)  # shift left + append candidate
#     out_plus, _ = pact_model.pact({"state": state_seq, "action": action_plus})
#     state_ctx_plus, action_ctx_plus = out_plus[:,0::2,:], out_plus[:,1::2,:]
#     return action_ctx_plus[:,-1,:], state_ctx_plus, action_ctx_plus

# def critic_score(pact_model: PACTPolicyCritic,
#                  state_seq: torch.Tensor, action_seq: torch.Tensor, a_last: torch.Tensor) -> torch.Tensor:
#     """ hψ(s_t, a_last) scalar logit """
#     last_act_ctx, _, _ = append_action_and_encode(pact_model, state_seq, action_seq, a_last)
#     return pact_model.critic(last_act_ctx).squeeze(-1)  # (B,)

# def compute_nominal_action(obs: Dict[str, Any], action_low=None, action_high=None) -> np.ndarray:
#     """
#     Simple LQR-like nominal for double integrator in 2D (acceleration control).
#     Uses dt from env via velocities inside obs if present; if not, falls back to gains.
#     """
#     # Prefer direct state/goal if present
#     if "state" in obs and "goal" in obs and len(obs["state"]) >= 4 and len(obs["goal"]) >= 4:
#         x, y, vx, vy = obs["state"][:4]
#         gx, gy, gvx, gvy = obs["goal"][:4]
#         ex, ey, evx, evy = (x-gx), (y-gy), (vx-gvx), (vy-gvy)
#     elif "state_goal" in obs and len(obs["state_goal"]) >= 6:
#         x, y, vx, vy, dx, dy = obs["state_goal"][:6]
#         ex, ey, evx, evy = (x - (x-dx)), (y - (y-dy)), vx, vy  # goal vel ~ 0
#     else:
#         # last resort: assume first two entries are positions
#         v = obs.get("full_obs", np.zeros(38, dtype=np.float32))
#         x, y = v[0], v[1]
#         vx, vy = (v[2] if len(v)>2 else 0.0), (v[3] if len(v)>3 else 0.0)
#         # no goal -> zero
#         ex, ey, evx, evy = x, y, vx, vy

#     # PD controller as "nominal" (LQR-like)
#     kx, kv = 1.0, 0.8
#     ux = -kx*ex - kv*evx
#     uy = -kx*ey - kv*evy
#     u = np.array([ux, uy], dtype=np.float32)
#     if action_low is not None and action_high is not None:
#         u = np.clip(u, action_low, action_high)
#     return u

# def rectify_action_linearized(pact_model: PACTPolicyCritic,
#                               state_seq: torch.Tensor, action_seq: torch.Tensor,
#                               a_pred: torch.Tensor, margin: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     One-constraint least-change step:
#       if h < margin:  a_safe = a_pred + ((margin - h) / (||∇h||^2 + eps)) * ∇h
#     Returns a_safe (detached), h (detached), grad (detached).
#     """
#     a_pred = a_pred.detach().requires_grad_(True)
#     h = critic_score(pact_model, state_seq, action_seq, a_pred)  # (B,)
#     g = torch.autograd.grad(h.sum(), a_pred, retain_graph=False, create_graph=False)[0]  # (B,A)
#     viol = (margin - h).clamp_min(0.0).unsqueeze(-1)  # (B,1)
#     denom = (g*g).sum(dim=-1, keepdim=True) + eps
#     delta = viol * g / denom
#     a_safe = a_pred + delta
#     return a_safe.detach(), h.detach(), g.detach()


# # -----------------------------
# # Online runner with buffers and training
# # -----------------------------
# class OnlineRunner:
#     def __init__(self, cfg: OnlineConfig):
#         self.cfg = cfg
#         self.device = torch.device(cfg.device)

#         # Build env
#         base = DoubleIntegrator(num_agents=1, area_size=cfg.area_size, max_step=cfg.max_step, dt=cfg.dt)
#         self.env = SingleAgentDIEnv(base, include_velocity=True, normalize_lidar=True, full_obs_keys=("state_goal","lidar"))
#         obs = self.env.reset(seed=cfg.seed)
#         full_obs = np.asarray(obs["full_obs"], dtype=np.float32)
#         S = int(full_obs.shape[0])
#         A = 2  # acceleration in x,y

#         # Build PACT policy+critic
#         self.model = PACTPolicyCritic(state_dim=S, action_dim=A, ctx_tokens=cfg.ctx_tokens,
#                                       n_embd=cfg.n_embd, n_layer=cfg.n_layer, n_head=cfg.n_head,
#                                       action_input_type="continuous").to(self.device)
#         self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

#         # Buffers
#         cap = cfg.buffer_capacity
#         self.primary = RingBuffer(cap); self.safe_buf = RingBuffer(cap); self.unsafe_buf = RingBuffer(cap)

#         # Rolling context
#         T = cfg.ctx_tokens // 2
#         self.ctx = RollingContext(T=T, S=S, A=A, device=cfg.device)
#         # Init actions to zeros
#         self.ctx.init(s0=full_obs, a0=np.zeros((A,), dtype=np.float32))

#         # Action limits (from env if present)
#         self.act_low  = obs.get("action_low",  np.array([-cfg.action_clip, -cfg.action_clip], dtype=np.float32))
#         self.act_high = obs.get("action_high", np.array([ cfg.action_clip,  cfg.action_clip], dtype=np.float32))

#     def step_once(self, t:int) -> Dict[str, Any]:
#         cfg = self.cfg
#         device = self.device

#         # Build window
#         state_seq_np, action_seq_np = self.ctx._states.copy(), self.ctx._actions.copy()
#         state_seq  = torch.as_tensor(state_seq_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,S)
#         action_seq = torch.as_tensor(action_seq_np, dtype=torch.float32, device=device).unsqueeze(0) # (1,T,A)

#         # Nominal
#         obs = self.env._last_obs if hasattr(self.env, "_last_obs") else None
#         if obs is None:
#             # after reset(), env.reset returns obs; keep it
#             obs = self.env.reset(seed=self.cfg.seed + t)
#         u_nom = compute_nominal_action(obs, self.act_low, self.act_high)   # (2,)

#         # Policy: Δu from last state embedding
#         with torch.no_grad():
#             out, _ = self.model.pact({"state": state_seq, "action": action_seq})
#             state_ctx = out[:,0::2,:]
#             last_state = state_ctx[:,-1,:]
#             delta = self.model.policy(last_state)   # (1,2)
#             u_pred = torch.as_tensor(u_nom, device=device).unsqueeze(0) + delta
#             u_pred = torch.clamp(u_pred, torch.as_tensor(self.act_low, device=device), torch.as_tensor(self.act_high, device=device))

#         # Critic score at predicted action + rectification (no horizon)
#         a_safe, h_pred, g_pred = rectify_action_linearized(self.model, state_seq, action_seq, u_pred, margin=cfg.margin, eps=cfg.grad_eps)
#         h_exec = critic_score(self.model, state_seq, action_seq, a_safe).detach()

#         # Execute
#         a_exec_np = a_safe.squeeze(0).cpu().numpy()
#         next_obs, reward, done, info = self.env.step(a_exec_np)

#         # Label (one-step)
#         label_safe = 0
#         # Prefer explicit flags; fallback to cost/done
#         if isinstance(info, dict):
#             if info.get("inside_obstacles") or info.get("unsafe") or info.get("collision"):
#                 label_safe = 0
#             else:
#                 label_safe = 1
#         else:
#             label_safe = 1 if not done else 0

#         # Update context
#         s_next = np.asarray(next_obs["full_obs"], dtype=np.float32)
#         self.ctx.update_after_step(s_next, a_exec_np)

#         # Store
#         item = ReplayItem(
#             state_seq=state_seq_np,
#             action_seq=action_seq_np,
#             u_nom=u_nom.astype(np.float32),
#             u_pred=u_pred.detach().cpu().numpy().squeeze(0).astype(np.float32),
#             u_exec=a_exec_np.astype(np.float32),
#             label_safe=int(label_safe),
#             h_pred=float(h_pred.item()),
#             h_exec=float(h_exec.item())
#         )
#         self.primary.push(item)
#         (self.safe_buf if label_safe==1 else self.unsafe_buf).push(item)

#         return {"t": t, "reward": reward, "done": done, "label_safe": label_safe,
#                 "h_pred": float(h_pred.item()), "h_exec": float(h_exec.item())}

#     def sample_minibatch(self, batch_size:int) -> Dict[str, torch.Tensor]:
#         # Merge some safe + some unsafe to avoid skew
#         n_unsafe = min(len(self.unsafe_buf), batch_size//2)
#         n_safe   = batch_size - n_unsafe
#         items = []
#         if n_safe>0 and len(self.safe_buf)>0:
#             items += self.safe_buf.sample(n_safe)
#         if n_unsafe>0 and len(self.unsafe_buf)>0:
#             items += self.unsafe_buf.sample(n_unsafe)
#         if len(items)==0:
#             items = self.primary.sample(batch_size)

#         # to tensors
#         T = self.cfg.ctx_tokens // 2
#         S = items[0].state_seq.shape[-1]
#         A = items[0].action_seq.shape[-1]
#         B = len(items)

#         st = torch.as_tensor(np.stack([it.state_seq for it in items], axis=0), dtype=torch.float32, device=self.device)
#         ac = torch.as_tensor(np.stack([it.action_seq for it in items], axis=0), dtype=torch.float32, device=self.device)
#         u_nom = torch.as_tensor(np.stack([it.u_nom for it in items], axis=0), dtype=torch.float32, device=self.device)
#         u_exec= torch.as_tensor(np.stack([it.u_exec for it in items], axis=0), dtype=torch.float32, device=self.device)
#         u_pred_cached = torch.as_tensor(np.stack([it.u_pred for it in items], axis=0), dtype=torch.float32, device=self.device)
#         y = torch.as_tensor(np.array([it.label_safe for it in items], dtype=np.float32), device=self.device)

#         return {"state_seq": st, "action_seq": ac, "u_nom": u_nom, "u_exec": u_exec, "u_pred_cached": u_pred_cached, "y": y}

#     def train_step(self) -> Dict[str, float]:
#         if len(self.primary) < 4:
#             return {"loss": 0.0}

#         batch = self.sample_minibatch(self.cfg.batch_size)
#         st, ac, u_nom, u_exec, y = batch["state_seq"], batch["action_seq"], batch["u_nom"], batch["u_exec"], batch["y"]

#         # Forward backbone
#         out, _ = self.model.pact({"state": st, "action": ac})
#         state_ctx = out[:,0::2,:]; action_ctx = out[:,1::2,:]
#         last_state = state_ctx[:,-1,:]

#         # Policy prediction
#         delta = self.model.policy(last_state)               # (B,2)
#         u_pred = torch.clamp(u_nom + delta,
#                              min=torch.as_tensor(self.act_low,  device=self.device),
#                              max=torch.as_tensor(self.act_high, device=self.device))

#         # Critic on u_pred and u_exec
#         h_pred = critic_score(self.model, st, ac, u_pred)
#         h_exec = critic_score(self.model, st, ac, u_exec)

#         # --- Losses ---
#         # Critic: BCE (safe=1), margin hinge, gradient regularizer @ u_pred
#         L_bce = nn.functional.binary_cross_entropy_with_logits(h_pred, y)
#         L_margin = ((1-y) * torch.relu(self.cfg.margin + h_pred) + 
#                     (y)   * torch.relu(self.cfg.margin - h_pred)).mean()

#         # grad penalty (smooth rectifier)
#         u_pred_req = u_pred.detach().requires_grad_(True)
#         h_tmp = critic_score(self.model, st, ac, u_pred_req)
#         g = torch.autograd.grad(h_tmp.sum(), u_pred_req, retain_graph=False, create_graph=False)[0]
#         L_grad = (g*g).sum(dim=-1).mean()

#         # Policy loss (QP-mimic)
#         L_pi_safe = ((u_pred - u_nom)**2).sum(dim=-1) * y
#         L_pi_fix  = ((u_pred - u_exec)**2).sum(dim=-1) * (1.0 - y)
#         L_pi = self.cfg.w_pi_safe*L_pi_safe.mean() + self.cfg.w_pi_fix*L_pi_fix.mean()

#         loss = self.cfg.w_bce*L_bce + self.cfg.w_margin*L_margin + self.cfg.w_grad*L_grad + L_pi

#         self.opt.zero_grad(set_to_none=True)
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
#         self.opt.step()

#         return {"loss": float(loss.item()),
#                 "L_bce": float(L_bce.item()),
#                 "L_margin": float(L_margin.item()),
#                 "L_grad": float(L_grad.item()),
#                 "L_pi": float(L_pi.item())}

#     def run_episode(self, max_steps: Optional[int]=None) -> Dict[str, Any]:
#         max_steps = max_steps or self.cfg.max_step
#         obs = self.env.reset(seed=self.cfg.seed)
#         # (re)init context with the new observation
#         self.ctx.init(obs["full_obs"].astype(np.float32), np.zeros((2,), dtype=np.float32))

#         ep_reward = 0.0
#         logs = []
#         for t in range(max_steps):
#             step_out = self.step_once(t)
#             ep_reward += float(step_out["reward"])
#             if self.cfg.update_every>0 and (t % self.cfg.update_every == 0):
#                 train_out = self.train_step()
#             else:
#                 train_out = {"loss": 0.0}

#             logs.append({**step_out, **train_out})
#             if self.cfg.verbose and (t % 20 == 0):
#                 print(f"[t={t:03d}] r={step_out['reward']:.3f} h_pred={step_out['h_pred']:.3f} "
#                       f"h_exec={step_out['h_exec']:.3f} label={step_out['label_safe']} "
#                       f"loss={train_out.get('loss',0.0):.4f}")
#             if step_out["done"]:
#                 break

#         return {"ep_reward": ep_reward, "steps": t+1, "logs": logs}


# # -----------------------------
# # Convenience API
# # -----------------------------
# def run_online_episode(cfg: Optional[OnlineConfig]=None) -> Dict[str, Any]:
#     cfg = cfg or OnlineConfig()
#     runner = OnlineRunner(cfg)
#     return runner.run_episode()


# if __name__ == "__main__":
#     cfg = OnlineConfig()
#     out = run_online_episode(cfg)
#     print(f"Episode reward={out['ep_reward']:.3f} steps={out['steps']}")