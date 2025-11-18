"""
PACT Online Agent: End-to-End Online Training Framework (MPC-like CBF Look-Ahead Edition)

This script implements an online training loop for a transformer-based agent (PACT)
that learns to navigate safely in a 2D environment with obstacles. The agent uses:
- A transformer policy network to propose actions based on historical context
- A critic network (barrier function) to evaluate action safety
- A QP-based safety teacher to provide safe action corrections
- An MPC-like look-ahead using the black-box simulator to enforce discrete-time CBF trends
- An alpha-blending mechanism to gradually shift from teacher to learned policy

Key Features:
- Model-free action selection (no explicit dynamics model required)
- Online updates (learns while acting in the environment)
- Safety-critical design (Control Barrier Functions, discrete-time)
- Optional slack in the QP teacher for robustness
- Short-horizon CBF look-ahead (MPC-like) for adaptive, predictive safety
- Replay buffer for stable training
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import os
import numpy as np
import torch
import torch.nn as nn

from gcbfplus.env.base import RolloutResult  # type: ignore

# ---- Project Imports ----
# Import the PACT transformer model (policy + critic)
from src.models.modules.a import PACTPolicyCritic  # type: ignore

# Import the environment: DoubleIntegrator (physics) + SingleAgentDIEnv (wrapper)
from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv  # type: ignore
from gcbfplus.env.double_integrator import DoubleIntegrator  # type: ignore
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import argparse
import os

device = "cpu" # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Project Imports ----
# Import the PACT transformer model (policy + critic)
from src.models.modules.a import PACTPolicyCritic

# Import the environment: DoubleIntegrator (physics) + SingleAgentDIEnv (wrapper)
from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv
from gcbfplus.env.double_integrator import DoubleIntegrator


# # =============================
# # Configuration
# # =============================
@dataclass
class OnlineConfig:
    """
    Configuration dataclass for the online training framework.
    Contains all hyperparameters for the model, environment, and training process.
    """
    
    # === Transformer Model Architecture ===
    ctx_tokens: int = 16      # Total number of tokens in context window (state + action pairs)
    n_embd: int = 128         # Embedding dimension for transformer
    n_layer: int = 4          # Number of transformer layers
    n_head: int = 8           # Number of attention heads
    device: str = "cpu"       # Device to run on ("cpu" or "cuda"), will change it later, when all integration will be finished
    
    # === Environment Parameters ===
    area_size: float = 6.0    # Size of the square environment (from -area_size to +area_size)
    max_step: int = 256       # Maximum steps per episode
    dt: float = 0.03          # Timestep duration (seconds)
    seed: int = 0             # Random seed for reproducibility

    # === Safety Parameters ===
    # These control the QP-based safety teacher that corrects unsafe actions
    margin: float = 0.15            # Safety margin for the barrier function (h >= margin for safe)
    margin_boost: float = 0.3       # Additional margin used when a verification fails
    grad_eps: float = 1e-8          # Small epsilon for numerical stability in gradient calculations
    action_clip: float = 2.0        # Maximum action magnitude (force limits)

    # Optional slack for the QP teacher (soft projection)
    # If rho <= 0, the QP is "hard" (no slack). For robustness try rho in [10, 100].
    qp_soft_rho: float = 0.0

    # === Verification Parameters ===
    # These control the forward simulation used to verify action safety
    verify_horizon: int = 8         # Number of steps to simulate ahead when verifying an action
    max_verify_attempts: int = 2    # How many times to retry with stronger safety margins

    # === MPC-like CBF look-ahead (uses black-box simulator) ===
    mpc_like_horizon: int = 4           # H steps to peek ahead
    mpc_like_use_qp_first: bool = True  # First action uses u_qp (safer) else the candidate itself
    mpc_like_repeat: str = "exec"       # {"exec","pi","qp","nom"} action repeated for steps 2..H
    mpc_like_softmin_tau: float = 5.0   # >0 soft-min aggregation; 0 => hard min
    w_cbf_hor_online: float = 0.0       # >0 enables a lightweight online gradient step from horizon CBF loss

    # === Alpha Blending (Teacher-to-Policy Transition) ===
    # Controls the gradual shift from relying on the QP teacher to trusting the learned policy
    alpha_init: float = 0.05        # Initial alpha (0 = full teacher, 1 = full policy)
    alpha_final: float = 0.90       # Final alpha after warmup
    alpha_warmup: int = 20000       # Number of steps over which alpha increases from init to final
    teacher_anchor: str = "nominal" # Anchor point for QP teacher: "nominal" or "policy"

    # === Training & Optimization ===
    lr: float = 3e-4                # Learning rate for Adam optimizer
    weight_decay: float = 1e-4      # L2 regularization weight
    grad_clip: float = 1.0          # Gradient clipping threshold (prevents exploding gradients)

    # === Loss Function Weights ===
    # These control the relative importance of each loss component
    w_bce: float = 1.0          # Binary cross-entropy loss (critic classification)
    w_margin: float = 0.5       # Margin loss (enforces confident predictions)
    w_grad: float = 1e-4        # Gradient penalty (smoothness of barrier function)
    w_pi_safe: float = 0.05     # Policy loss weight for safe state behavior
    w_exec: float = 0.10        # Execution consistency loss (stabilizes training distribution)
    w_hdot: float = 0.10        # Barrier derivative loss (enforces CBF condition)

    # === CBF Derivative Constraint ===
    hdot_lambda: float = 1.0    # Lambda parameter for discrete-time CBF: h_dot + lambda*h >= 0

    # === Online Update Schedule ===
    update_every: int = 1           # How often to perform a training update (every N steps)
    batch_size: int = 64            # Number of samples per training batch
    buffer_capacity: int = 20000    # Maximum size of the replay buffer

    # === Debugging ===
    verbose: bool = True            # Whether to print detailed logs during training

# # =============================
# # Replay Buffer Data Structures
# # =============================
class ReplayItem:
    """
    A single experience item stored in the replay buffer.
    
    Contains all information needed to train the agent from one timestep:
    - Current and next state/action windows (for the transformer context)
    - All action variants (nominal, policy, QP, executed)
    - Safety label and barrier values (for computing losses)
    """
    __slots__ = (
        "state_seq",        # Current state window (T, S) - input to transformer at time t
        "action_seq",       # Current action window (T, A) - input to transformer at time t
        "state_seq_next",   # Next state window (T, S) - for computing h(t+1) in CBF constraint
        "action_seq_next",  # Next action window (T, A) - for computing h(t+1) in CBF constraint
        "u_nom",            # Nominal action from LQR controller (goal-seeking, no safety)
        "u_pi",             # Policy action from transformer (learned behavior)
        "u_qp",             # QP teacher action (safety-corrected)
        "u_exec",           # Actually executed action (blend of QP and policy)
        "alpha",            # Blending coefficient used at this timestep
        "label_safe",       # Ground truth safety label: 1=safe, 0=unsafe (collision)
        "h_t",              # Barrier value at time t
        "h_tp1"             # Barrier value at time t+1 (for CBF derivative constraint)
    )
    
    def __init__(self, **kw):
        """Initialize by setting all provided keyword arguments as attributes."""
        for k, v in kw.items():
            setattr(self, k, v)


class RingBuffer:
    """
    Circular replay buffer for storing agent experiences.
    
    Implements a fixed-size buffer that overwrites the oldest data when full.
    Supports random sampling for training batches.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize an empty buffer with a given capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = int(capacity)
        self.data: List[ReplayItem] = []  # Storage for experiences
        self.ptr = 0                      # Write pointer for circular overwriting
    
    def __len__(self):
        """Return the current number of items in the buffer."""
        return len(self.data)
    
    def push(self, item: ReplayItem):
        """
        Add a new experience to the buffer.
        
        If the buffer is full, overwrites the oldest item.
        
        Args:
            item: ReplayItem to store
        """
        if len(self.data) < self.capacity:
            # Buffer not full yet - just append
            self.data.append(item)
        else:
            # Buffer full - overwrite oldest item and advance pointer
            self.data[self.ptr] = item
            self.ptr = (self.ptr + 1) % self.capacity
    
    def sample(self, n: int) -> List[ReplayItem]:
        """
        Randomly sample n experiences from the buffer.
        
        Args:
            n: Number of samples to draw
            
        Returns:
            List of n ReplayItems (or fewer if buffer is smaller than n)
        """
        n = min(n, len(self.data))
        if n == 0:
            return []
        # Sample without replacement
        idx = np.random.choice(len(self.data), size=n, replace=False)
        return [self.data[i] for i in idx]


# # =============================
# # Rolling Context Window
# # =============================
class RollingContext:
    """
    Manages the fixed-size historical context window for the transformer.
    
    The transformer needs to see the last T timesteps of (state, action) pairs.
    This class maintains a sliding window that:
    - Starts by repeating the initial state/action
    - Shifts left and adds new data as the agent acts
    - Provides efficient conversion to PyTorch tensors for model input
    
        States:  [s0, s0, s0]  ← Initial state repeated 3 times
        Actions: [a0, a0, a0]  ← Zero action repeated 3 times
        Step 1 (After first action):
        States:  [s0, s0, s1]  ← Shifted left, added new state s1
        Actions: [a0, a0, a1]  ← Shifted left, added executed action a1
        Step 2 (After second action):
        States:  [s0, s1, s2]  ← Shifted left, added new state s2
        Actions: [a0, a1, a2]  ← Shifted left, added executed action a2
        Step 3 (After third action):
        States:  [s1, s2, s3]  ← s0 "falls off" the left end
        Actions: [a1, a2, a3]  ← a0 "falls off" the left end
    """
    
    def __init__(self, T: int, S: int, A: int):
        """
        Initialize an empty context window.
        
        Args:
            T: Context length (number of timesteps to remember)
            S: State dimension
            A: Action dimension
        """
        self.T = T  # Window length
        self.S = S  # State dimension
        self.A = A  # Action dimension
        self._states = None   # Will be (T, S) numpy array
        self._actions = None  # Will be (T, A) numpy array
    
    def init(self, s0: np.ndarray, a0: Optional[np.ndarray] = None):
        """
        Initialize the window by repeating the initial state and action T times.
        
        This is done at the start of an episode to create a valid input for the transformer.
        
        Args:
            s0: Initial state (S,)
            a0: Initial action (A,), defaults to zero if not provided
        """
        # s0 = np.asarray(s0, dtype=np.float32)
        # if a0 is None:
        #     a0 = np.zeros((self.A,), dtype=np.float32)
        # a0 = np.asarray(a0, dtype=np.float32)
        # # Repeat the initial state and action T times to fill the window
        # self._states = np.repeat(s0[None, :], self.T, axis=0)   # (T, S)
        # self._actions = np.repeat(a0[None, :], self.T, axis=0)  # (T, A)
        s0 = np.asarray(s0, dtype=np.float32)
        if a0 is None:
            a0 = np.zeros((self.A,), dtype=np.float32)
        a0 = np.asarray(a0, dtype=np.float32)
        self._states = np.repeat(s0[None, :], self.T, axis=0)   # (T, S)
        self._actions = np.repeat(a0[None, :], self.T, axis=0)  # (T, A)
    
    def as_batch(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the current window to PyTorch tensors with batch dimension.
        
        Args:
            device: PyTorch device to place tensors on
            
        Returns:
            Tuple of (state_seq, action_seq) both with shape (1, T, D)
        """
        s = torch.as_tensor(self._states, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, S)
        a = torch.as_tensor(self._actions, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, A)
        return s, a
    
    def after_step_versions(self, s_next: np.ndarray, a_exec: np.ndarray):
        """
        Compute what the window would look like after taking a step.
        
        This is used for computing h(t+1) in the CBF derivative constraint.
        Does NOT modify the internal state.
        
        Args:
            s_next: Next state observed after taking action (S,)
            a_exec: Action that was executed (A,)
            
        Returns:
            Tuple of (next_states, next_actions) both with shape (T, D)
        """
        # Shift the window left by 1 and append the new data
        ns = np.concatenate([self._states[1:], s_next[None, :]], axis=0)   # (T, S)
        na = np.concatenate([self._actions[1:], a_exec[None, :]], axis=0)  # (T, A)
        return ns, na
    
    def update(self, s_next: np.ndarray, a_exec: np.ndarray):
        """
        Update the internal window by shifting left and adding new data.
        
        This is called after each environment step to maintain the rolling context.
        
        Args:
            s_next: Next state observed after taking action (S,)
            a_exec: Action that was executed (A,)
        """
        self._states, self._actions = self.after_step_versions(s_next, a_exec)


# =============================
# Safety Critic & QP Teacher
# =============================
def h_score(model: PACTPolicyCritic,
            state_seq: torch.Tensor, action_seq: torch.Tensor, a_last: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the barrier function (critic) for a given action.
    
    The critic looks at a historical context (state_seq, action_seq) and predicts
    the safety of adding action a_last to the end of the sequence.
    
    Process:
    1. Shift the action sequence left by 1 timestep
    2. Append a_last to the end
    3. Pass through the transformer to get embeddings
    4. Extract the final action token embedding
    5. Pass through the critic head to get a scalar safety score
    
    Args:
        model: PACTPolicyCritic (transformer + critic head)
        state_seq: Historical states (B, T, S)
        action_seq: Historical actions (B, T, A)
        a_last: Action to evaluate (B, A)
    
    Returns:
        Safety scores h(s, a) for each sample in the batch (B,)
        - h > 0 means safe
        - h < 0 means unsafe
    """
    B, T, A = action_seq.shape
    a_last = a_last.view(B, 1, A)
    # Create new action sequence: [action_seq[1:], a_last]
    action_plus = torch.cat([action_seq[:, 1:, :], a_last], dim=1)  # (B, T, A)
    
    # Forward pass through transformer
    out_plus, _ = model.pact({"state": state_seq, "action": action_plus})
    
    # Extract action tokens (transformer outputs tokens for both state and action)
    # Convention: out[:, 0::2] = state tokens, out[:, 1::2] = action tokens
    action_ctx_plus = out_plus[:, 1::2, :]  # (B, T, emb_dim)
    
    # Pass the final action token through the critic head
    return model.critic(action_ctx_plus[:, -1, :]).squeeze(-1)  # (B,)


# def qp_teacher_anchor_nominal(model: PACTPolicyCritic,
#                               state_seq: torch.Tensor,
#                               action_seq: torch.Tensor,
#                               u_nom: torch.Tensor,
#                               margin: float,
#                               eps: float = 1e-8) -> torch.Tensor:
#     """
#     Quadratic Programming (QP) based safety teacher.
    
#     Solves the optimization problem:
#         minimize:   0.5 * ||u - u_nom||^2
#         subject to: h(u0) + g^T(u - u0) >= margin
    
#     Where:
#     - u_nom is the nominal (goal-seeking) action
#     - h is the barrier function (critic)
#     - g is the gradient of h with respect to the action
#     - margin is the safety threshold
    
#     This finds the action closest to u_nom that satisfies the safety constraint.
#     The constraint is linearized around u0 = u_nom for computational efficiency.
    
#     Mathematical derivation:
#     1. The Lagrangian is: L = 0.5||u - u_nom||^2 - λ(h(u0) + g^T(u - u0) - margin)
#     2. Setting ∇_u L = 0 gives: u - u_nom - λg = 0, so u = u_nom + λg
#     3. The KKT complementarity condition determines λ
#     4. If the constraint is inactive (already safe), λ=0 and u = u_nom
#     5. If active, λ is solved from the constraint: g^T(u_nom + λg) = margin - h(u0) + g^T u0
    
#     example : 
    
#     Args:
#         model: PACTPolicyCritic network
#         state_seq: Historical states (B, T, S)
#         action_seq: Historical actions (B, T, A)
#         u_nom: Nominal actions to anchor the QP (B, A)
#         margin: Safety margin (h must be >= margin)
#         eps: Small constant for numerical stability
    
#     Returns:
#         Safe actions u_qp (B, A) that are closest to u_nom while respecting safety

#     Example:
#     (u_nom=[0,0])
#     critic says (h(u_nom)=-0.2) (unsafe)
#     desired margin (=0.1) → we need to gain (0.1 - (-0.2)=0.3) of safety
#     gradient (g=[2,,1])

#     compute pieces:

#     |g|^2 = 2^2 + 1^2 = 4 + 1 = 5 
#     lambda = (0.3) / 5 = 0.06   (since gap is (0.3))

#     new action: ( u_qp = [0,0] + 0.06*[2,1] = [0.12,0.06] )

#     sanity check (does the linearized constraint hit the margin?):
#     (g^T u_qp = 2* 0.12 + 1* 0.06 = 0.24 + 0.06 = 0.30)

#     linearized score at (u_qp): (h(u_nom) + g^T(u_{qp}-u_{nom}) = -0.2 + 0.30 = 0.10 = {margin}) 

#     already safe case: if (h(u_{nom})=0.2) and margin (=0.1) → gap (= -0.1) → (lambda=0) → (u_{qp}=u_{nom}).
#     """
#     # Evaluate h and its gradient at u0 = u_nom
#     u0 = u_nom.detach().clone().requires_grad_(True)  # (B, A)
    
#     # Compute h(u0) for each sample in the batch
#     h_per = h_score(model, state_seq, action_seq, u0)  # (B,)
#     h_sum = h_per.sum()  # Sum for backprop
    
#     # Compute gradient g = ∂h/∂u at u = u0
#     g = torch.autograd.grad(h_sum, u0, retain_graph=False, create_graph=False)[0]  # (B, A)

#     # Define the safety constraint as a half-space: g^T u >= b
#     # where b = margin - h(u0) + g^T u0
#     b = (margin - h_per).unsqueeze(-1) + (g * u0).sum(-1, keepdim=True)  # (B, 1)
    
#     # Denominator for projection: ||g||^2
#     denom = (g * g).sum(-1, keepdim=True).clamp_min(eps)  # (B, 1)
    
#     # Check if u_nom violates the constraint
#     gt_unom = (g * u0).sum(-1, keepdim=True)  # (B, 1) = g^T u_nom
    
#     # Lagrange multiplier λ (KKT condition)
#     # If λ > 0, the constraint is active and we project u_nom onto the boundary
#     # If λ ≤ 0, u_nom already satisfies the constraint, so use it as-is
#     lam = ((b - gt_unom) / denom).clamp_min(0.0)  # (B, 1)
    
#     # Final safe action: u_qp = u_nom + λ * g
#     u_qp = u0 + lam * g  # (B, A)
    
#     return u_qp.detach()  # Detach to prevent gradients flowing back through QP
def qp_teacher_anchor_nominal(model: PACTPolicyCritic,
                              state_seq: torch.Tensor,
                              action_seq: torch.Tensor,
                              u_nom: torch.Tensor,
                              margin: float,
                              eps: float = 1e-8,
                              rho: float = 0.0) -> torch.Tensor:
    """
    QP-based safety teacher (closed-form projection onto a linearized half-space).
    Hard QP:
        minimize:   0.5 * ||u - u_nom||^2
        subject to: h(u0) + g^T(u - u0) >= margin
    Soft QP (if rho>0): allow slack ξ with quadratic penalty; closed-form denominator += 1/rho.

    Returns:
        Safe actions u_qp (B, A), detached from autograd graph.
    """
    # Anchor at u0 = u_nom and enable gradients to get g
    u0 = u_nom.detach().clone().requires_grad_(True)  # (B, A)
    h_per = h_score(model, state_seq, action_seq, u0)  # (B,)
    g = torch.autograd.grad(h_per.sum(), u0, retain_graph=False, create_graph=False)[0]  # (B, A)

    # Half-space g^T u >= b, where b = margin - h(u0) + g^T u0
    b = (margin - h_per).unsqueeze(-1) + (g * u0).sum(-1, keepdim=True)  # (B, 1)
    gt_unom = (g * u0).sum(-1, keepdim=True)                              # (B, 1)
    denom = (g * g).sum(-1, keepdim=True).clamp_min(eps)                  # (B, 1)

    if rho is None or rho <= 0.0:
        lam = ((b - gt_unom) / denom).clamp_min(0.0)  # hard
    else:
        lam = ((b - gt_unom) / (denom + 1.0/float(rho))).clamp_min(0.0)  # soft

    u_qp = u0 + lam * g  # (B, A)
    return u_qp.detach()

# # =============================
# # Online Training Runner
# # =============================
class OnlineRunner:
#     """
#     Main class that orchestrates the online training loop.
    
#     Responsibilities:
#     1. Initialize and manage the environment
#     2. Initialize and manage the transformer model
#     3. Maintain replay buffers for training
#     4. Execute the agent-environment interaction loop
#     5. Trigger training updates at regular intervals
#     6. Blend actions from the policy and safety teacher
    
#     The training is "online" because the agent learns continuously as it interacts
#     with the environment, rather than collecting data first and then training.
#     """
    
    def __init__(self, cfg: OnlineConfig):
        """
        Initialize the online training runner.
        
        Args:
            cfg: Configuration object with all hyperparameters
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # === Environment Setup ===
        # Create the physics simulator (DoubleIntegrator)
        base = DoubleIntegrator(
            num_agents=1, 
            area_size=cfg.area_size, 
            max_step=cfg.max_step, 
            dt=cfg.dt
        )
        # Wrap with SingleAgentDIEnv to get a clean Gym-like interface
        self.env = SingleAgentDIEnv(
            base, 
            include_velocity=True,           # Include velocity in state observation
            normalize_lidar=True,            # Normalize LiDAR readings to [0, 1]
            full_obs_keys=("state_goal", "lidar")  # Components to include in observation
        )
        
        # Reset environment to get initial observation and determine dimensions
        obs = self.env.reset(seed=cfg.seed)
        full_obs = np.asarray(obs["full_obs"], dtype=np.float32)
        S = int(full_obs.shape[0])  # State dimension
        A = 2                        # Action dimension (fx, fy)

        # === Model Setup ===
        # Create the PACT transformer model (policy + critic)
        self.model = PACTPolicyCritic(
            state_dim=S, 
            action_dim=A, 
            ctx_tokens=cfg.ctx_tokens,  # Total context length (state + action pairs)
            n_embd=cfg.n_embd,          # Embedding dimension
            n_layer=cfg.n_layer,        # Number of transformer layers
            n_head=cfg.n_head,          # Number of attention heads
            action_input_type="continuous"  # Continuous action space
        ).to(self.device)
        
        # Adam optimizer with weight decay (L2 regularization)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )

        # === Replay Buffers ===
        # We maintain 3 buffers to ensure balanced sampling of safe/unsafe experiences
        cap = cfg.buffer_capacity
        self.primary = RingBuffer(cap)    # All experiences
        self.safe_buf = RingBuffer(cap)   # Only safe experiences
        self.unsafe_buf = RingBuffer(cap) # Only unsafe experiences

        # === Rolling Context Window ===
        # The transformer needs to see T timesteps of history
        # Since each timestep has 2 tokens (state + action), ctx_tokens = 2*T
        T = cfg.ctx_tokens // 2
        self.ctx = RollingContext(T=T, S=S, A=A)
        # Initialize with the starting state and zero action
        self.ctx.init(s0=full_obs, a0=np.zeros((A,), dtype=np.float32))

        # Bounds
        self.act_low  = obs.get("action_low",  np.array([-cfg.action_clip, -cfg.action_clip], dtype=np.float32))
        self.act_high = obs.get("action_high", np.array([ cfg.action_clip,  cfg.action_clip], dtype=np.float32))

        self.global_step = 0

#     # ---- helpers ----
    def alpha(self) -> float:
        if self.cfg.alpha_warmup <= 0: return float(self.cfg.alpha_final)
        r = min(1.0, self.global_step / float(self.cfg.alpha_warmup))
        return float(self.cfg.alpha_init + r * (self.cfg.alpha_final - self.cfg.alpha_init))

    def nominal_action(self) -> np.ndarray:
        if hasattr(self.env, "nominal_action"):
            u = self.env.nominal_action()
            return np.asarray(u, dtype=np.float32).reshape(-1)
        return np.zeros((2,), dtype=np.float32)

    def verify_horizon(self, a0: np.ndarray, H: int) -> bool:
        if not hasattr(self.env, "rollout_sequence"):
            return True
        cand = np.tile(a0.reshape(1,-1), (H,1)).astype(np.float32)
        states, rewards, infos = self.env.rollout_sequence(cand, early_stop_on_violation=False)
        for k in range(len(infos)):
            inf = infos[k] if isinstance(infos, (list,tuple)) else infos
            if isinstance(inf, dict) and (inf.get("inside_obstacles") or inf.get("unsafe") or inf.get("collision")):
                return False
        return True
    
    # ---- CBF helpers ----
    def _cbf_residual(self, h_now: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        dt = max(self.cfg.dt, 1e-6)
        lam = self.cfg.hdot_lambda
        return lam * h_now + (h_next - h_now) / dt  # shape: (B,)


    @torch.no_grad()
    def cbf_rollout_score(self,
                          state_seq: torch.Tensor,
                          action_seq: torch.Tensor,
                          u_first: torch.Tensor,
                          u_pi: torch.Tensor,
                          u_qp: torch.Tensor,
                          u_nom: torch.Tensor,
                          H: int) -> Dict[str, Any]:
        """
        MPC-like "peek": simulate H steps open-loop and compute discrete CBF residuals r_i.
        Returns dict with residual list and an aggregate score (soft-min).
        """
        dev = self.device
        if H <= 0 or not hasattr(self.env, "rollout_sequence"):
            return {"residuals": [], "score": float("inf"), "h": []}

        # Select the action to repeat after the first step
        rep = self.cfg.mpc_like_repeat
        if rep == "pi":
            u_rep = u_pi
        elif rep == "qp":
            u_rep = u_qp
        elif rep == "nom":
            u_rep = u_nom
        else:  # "exec"
            u_rep = (1.0 - torch.tensor(self.alpha(), device=dev)) * u_qp + torch.tensor(self.alpha(), device=dev) * u_pi

        # Build candidate sequence for peek
        first = (u_qp if self.cfg.mpc_like_use_qp_first else u_first).squeeze(0).cpu().numpy()
        rep_np = u_rep.squeeze(0).cpu().numpy()
        U = np.vstack([first] + [rep_np for _ in range(max(0, H-1))]).astype(np.float32)  # (H,A)

        # Query simulator (does not advance real env)
        states, _, _ = self.env.rollout_sequence(U, early_stop_on_violation=False)
        states = np.asarray(states, dtype=np.float32)  # (H,S)

        # Copy windows
        win_s = self.ctx._states.copy()
        win_a = self.ctx._actions.copy()

        residuals = []
        hs = []

        # h_0 under first action
        h_curr = h_score(self.model, state_seq, action_seq,
                         torch.as_tensor(U[0][None, :], device=dev, dtype=torch.float32)).item()

        for i in range(H):
            s_next = states[i]; a_i = U[i]
            win_s = np.concatenate([win_s[1:], s_next[None, :]], axis=0)
            win_a = np.concatenate([win_a[1:], a_i[None, :]],  axis=0)

            st = torch.as_tensor(win_s[None, ...], dtype=torch.float32, device=dev)
            ac = torch.as_tensor(win_a[None, ...], dtype=torch.float32, device=dev)
            a_next = torch.as_tensor(U[i if i < H-1 else i], dtype=torch.float32, device=dev).view(1, -1)

            h_next = h_score(self.model, st, ac, a_next).item()
            hs.append(h_curr)
            r_i = self._cbf_residual(torch.tensor(h_curr, device=dev), torch.tensor(h_next, device=dev)).item()
            residuals.append(float(r_i))
            h_curr = h_next

        # Aggregate residuals using soft-min (higher is safer)
        if self.cfg.mpc_like_softmin_tau > 0:
            tau = self.cfg.mpc_like_softmin_tau
            score = float(-tau * torch.logsumexp(-torch.tensor(residuals, device=dev)/tau, dim=0))
        else:
            score = float(min(residuals))

        return {"residuals": residuals, "score": score, "h": hs}

    def cbf_rollout_loss(self,
                         state_seq: torch.Tensor,
                         action_seq: torch.Tensor,
                         U: np.ndarray) -> torch.Tensor:
        """
        Differentiable ONLINE auxiliary loss: mean ReLU(-r_i) over the peeked horizon.
        Env provides future states (constants); gradients flow through h(.) wrt model params.
        """
        dev = self.device
        if not hasattr(self.env, "rollout_sequence") or len(U) == 0:
            return torch.tensor(0.0, device=dev)

        states, _, _ = self.env.rollout_sequence(U.astype(np.float32), early_stop_on_violation=False)
        states = np.asarray(states, dtype=np.float32)

        win_s = self.ctx._states.copy()
        win_a = self.ctx._actions.copy()

        a0 = torch.as_tensor(U[0][None, :], dtype=torch.float32, device=dev)
        h_curr = h_score(self.model, state_seq, action_seq, a0)  # (1,)
        penalties = []

        for i in range(len(U)):
            s_next = states[i]; a_i = U[i]
            win_s = np.concatenate([win_s[1:], s_next[None, :]], axis=0)
            win_a = np.concatenate([win_a[1:], a_i[None, :]],  axis=0)

            st = torch.as_tensor(win_s[None, ...], dtype=torch.float32, device=dev)
            ac = torch.as_tensor(win_a[None, ...], dtype=torch.float32, device=dev)
            a_next = torch.as_tensor(U[i if i < len(U)-1 else i][None, :], dtype=torch.float32, device=dev)

            h_next = h_score(self.model, st, ac, a_next)  # (1,)
            r_i = self._cbf_residual(h_curr, h_next)      # (1,)
            penalties.append(torch.relu(-r_i))
            h_curr = h_next

        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=dev)

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

        # QP teacher anchored at nominal (optionally soft)
        u_qp = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom,
                                         margin=cfg.margin, eps=cfg.grad_eps, rho=cfg.qp_soft_rho)

        # Alpha-mixed action
        alpha = self.alpha()
        u_exec = (1.0 - alpha)*u_qp + alpha*u_pi
        u_exec = torch.clamp(u_exec,
                             min=torch.as_tensor(self.act_low,  device=dev),
                             max=torch.as_tensor(self.act_high, device=dev))

        # === MPC-like CBF look-ahead selection ===
        H = int(cfg.mpc_like_horizon)
        cbf_diag = {}
        if H > 0 and hasattr(self.env, "rollout_sequence"):
            scores = {
                "exec": self.cbf_rollout_score(state_seq, action_seq, u_exec, u_pi, u_qp, u_nom, H),
                "qp":   self.cbf_rollout_score(state_seq, action_seq, u_qp,   u_pi, u_qp, u_nom, H),
                "pi":   self.cbf_rollout_score(state_seq, action_seq, u_pi,   u_pi, u_qp, u_nom, H),
            }
            choice = max(scores.keys(), key=lambda k: scores[k]["score"])
            u_chosen = {"exec": u_exec, "qp": u_qp, "pi": u_pi}[choice]
            cbf_diag = scores[choice]

            # If even the best horizon looks unsafe, tighten margin and retry QP
            if scores[choice]["score"] < 0.0 and cfg.max_verify_attempts > 0:
                u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom,
                                                  margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps, rho=cfg.qp_soft_rho)
                score2 = self.cbf_rollout_score(state_seq, action_seq, u_qp2, u_pi, u_qp2, u_nom, H)
                if score2["score"] > scores[choice]["score"]:
                    u_chosen = u_qp2
                    cbf_diag = score2

            u_exec = torch.clamp(u_chosen,
                                 min=torch.as_tensor(self.act_low,  device=dev),
                                 max=torch.as_tensor(self.act_high, device=dev))

            # Optional: online auxiliary gradient step to reduce horizon CBF violations
            if cfg.w_cbf_hor_online > 0.0:
                rep = cfg.mpc_like_repeat
                if rep == "pi":   u_rep = u_pi
                elif rep == "qp": u_rep = u_qp
                elif rep == "nom":u_rep = u_nom
                else:
                    u_rep = (1.0 - torch.tensor(alpha, device=dev)) * u_qp + torch.tensor(alpha, device=dev) * u_pi
                U = np.vstack([u_exec.squeeze(0).cpu().numpy()] +
                              [u_rep.squeeze(0).cpu().numpy() for _ in range(max(0, H-1))]).astype(np.float32)
                L_cbf_hor = self.cbf_rollout_loss(state_seq, action_seq, U) * float(cfg.w_cbf_hor_online)
                if L_cbf_hor.item() > 0.0:
                    self.opt.zero_grad(set_to_none=True)
                    L_cbf_hor.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    self.opt.step()

        # === Physics verification & repairs (final guard) ===
        a_try = u_exec.squeeze(0).cpu().numpy()
        ok = self.verify_horizon(a_try, cfg.verify_horizon)
        if not ok and cfg.max_verify_attempts > 0:
            u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom,
                                              margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps, rho=cfg.qp_soft_rho)
            u_exec2 = ((1.0 - alpha)*u_qp2 + alpha*u_pi).clamp(
                torch.as_tensor(self.act_low, device=dev),
                torch.as_tensor(self.act_high, device=dev)
            )
            a_try2 = u_exec2.squeeze(0).cpu().numpy()
            if self.verify_horizon(a_try2, cfg.verify_horizon):
                u_exec = u_exec2
            else:
                if self.verify_horizon(u_qp.squeeze(0).cpu().numpy(), cfg.verify_horizon):
                    u_exec = u_qp
                else:
                    if self.verify_horizon(u_nom.squeeze(0).cpu().numpy(), cfg.verify_horizon):
                        u_exec = u_nom

        # Critic reads
        with torch.no_grad():
            h_pi  = h_score(self.model, state_seq, action_seq, u_pi).item()
            h_qp  = h_score(self.model, state_seq, action_seq, u_qp).item()
            h_exe = h_score(self.model, state_seq, action_seq, u_exec).item()

        # Execute in real env
        a_exec_np = u_exec.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = self.env.step(a_exec_np)
        label_safe = 1
        if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
            label_safe = 0
        if done and label_safe == 1:
            label_safe = 0

        # Next-window versions (for one-step hdot loss)
        s_next = np.asarray(next_obs["full_obs"], dtype=np.float32)
        state_seq_next_np, action_seq_next_np = self.ctx.after_step_versions(s_next, a_exec_np)
        state_seq_next = torch.as_tensor(state_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)
        action_seq_next= torch.as_tensor(action_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)

        with torch.no_grad():
            h_tp1 = h_score(self.model, state_seq_next, action_seq_next, u_exec).item()

        # Update context
        self.ctx.update(s_next, a_exec_np)

        # Store
        item = ReplayItem(
            state_seq=state_seq.squeeze(0).cpu().numpy(),
            action_seq=action_seq.squeeze(0).cpu().numpy(),
            state_seq_next=state_seq_next.squeeze(0).cpu().numpy(),
            action_seq_next=action_seq_next.squeeze(0).cpu().numpy(),
            u_nom=u_nom.squeeze(0).cpu().numpy(),
            u_pi=u_pi.squeeze(0).cpu().numpy(),
            u_qp=u_qp.squeeze(0).cpu().numpy(),
            u_exec=a_exec_np.astype(np.float32),
            alpha=float(alpha),
            label_safe=int(label_safe),
            h_t=float(h_exe),
            h_tp1=float(h_tp1),
        )
        self.primary.push(item)
        (self.safe_buf if label_safe==1 else self.unsafe_buf).push(item)

        self.global_step += 1

        out = {"t": t, "reward": float(reward), "done": bool(done), "label_safe": int(label_safe),
               "h_pi": float(h_pi), "h_qp": float(h_qp), "h_exec": float(h_exe), "alpha": float(alpha)}
        if cbf_diag:
            out["cbf_min"] = float(min(cbf_diag.get("residuals", [0.0])))
            out["cbf_score"] = float(cbf_diag.get("score", 0.0))
        return out

#     # ---- one control step ----
#     def step_once(self, t:int) -> Dict[str, Any]:
#         cfg = self.cfg; dev = self.device

#         # Windows
#         state_seq, action_seq = self.ctx.as_batch(dev)  # (1,T,S), (1,T,A)

#         # Nominal LQR
#         u_nom_np = self.nominal_action()
#         u_nom = torch.as_tensor(u_nom_np, device=dev).unsqueeze(0)  # (1,2)

#         # Policy proposal
#         with torch.no_grad():
#             out, _ = self.model.pact({"state": state_seq, "action": action_seq})
#             last_state = out[:,0::2,:][:,-1,:]
#             delta = self.model.policy(last_state)  # (1,2)
#             u_pi = torch.clamp(u_nom + delta,
#                                min=torch.as_tensor(self.act_low,  device=dev),
#                                max=torch.as_tensor(self.act_high, device=dev))

#         # QP teacher anchored at nominal
#         u_qp = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom, margin=cfg.margin, eps=cfg.grad_eps)

#         # Alpha-mixed action
#         alpha = self.alpha()
#         u_exec = (1.0 - alpha)*u_qp + alpha*u_pi
#         u_exec = torch.clamp(u_exec,
#                              min=torch.as_tensor(self.act_low,  device=dev),
#                              max=torch.as_tensor(self.act_high, device=dev))

#         # Horizon verification & repairs
#         a_try = u_exec.squeeze(0).cpu().numpy()
#         ok = self.verify_horizon(a_try, cfg.verify_horizon)
#         if not ok and cfg.max_verify_attempts > 0:
#             # stronger margin
#             u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom, margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps)
#             u_exec2 = ((1.0 - alpha)*u_qp2 + alpha*u_pi).clamp(
#                 torch.as_tensor(self.act_low, device=dev),
#                 torch.as_tensor(self.act_high, device=dev)
#             )
#             a_try2 = u_exec2.squeeze(0).cpu().numpy()
#             if self.verify_horizon(a_try2, cfg.verify_horizon):
#                 u_exec = u_exec2
#             else:
#                 # fall back to pure QP if that passes
#                 if self.verify_horizon(u_qp.squeeze(0).cpu().numpy(), cfg.verify_horizon):
#                     u_exec = u_qp
#                 else:
#                     # last resort: nominal if preview says ok
#                     if self.verify_horizon(u_nom.squeeze(0).cpu().numpy(), cfg.verify_horizon):
#                         u_exec = u_nom

#         # Critic reads
#         with torch.no_grad():
#             h_pi  = h_score(self.model, state_seq, action_seq, u_pi).item()
#             h_qp  = h_score(self.model, state_seq, action_seq, u_qp).item()
#             h_exe = h_score(self.model, state_seq, action_seq, u_exec).item()

#         # Execute in real env
#         a_exec_np = u_exec.squeeze(0).cpu().numpy()
#         next_obs, reward, done, info = self.env.step(a_exec_np)
#         label_safe = 1
#         if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
#             label_safe = 0
#         if done and label_safe == 1:
#             # treat terminal as unsafe unless explicitly marked safe
#             label_safe = 0

#         # Next-window versions (for hdot)
#         s_next = np.asarray(next_obs["full_obs"], dtype=np.float32)
#         state_seq_next_np, action_seq_next_np = self.ctx.after_step_versions(s_next, a_exec_np)
#         state_seq_next = torch.as_tensor(state_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)
#         action_seq_next= torch.as_tensor(action_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)

#         with torch.no_grad():
#             h_tp1 = h_score(self.model, state_seq_next, action_seq_next, u_exec).item()

#         # Update context
#         self.ctx.update(s_next, a_exec_np)

#         # Store
#         item = ReplayItem(
#             state_seq=state_seq.squeeze(0).cpu().numpy(),
#             action_seq=action_seq.squeeze(0).cpu().numpy(),
#             state_seq_next=state_seq_next.squeeze(0).cpu().numpy(),
#             action_seq_next=action_seq_next.squeeze(0).cpu().numpy(),
#             u_nom=u_nom.squeeze(0).cpu().numpy(),
#             u_pi=u_pi.squeeze(0).cpu().numpy(),
#             u_qp=u_qp.squeeze(0).cpu().numpy(),
#             u_exec=a_exec_np.astype(np.float32),
#             alpha=float(alpha),
#             label_safe=int(label_safe),
#             h_t=float(h_exe),
#             h_tp1=float(h_tp1),
#         )
#         self.primary.push(item)
#         (self.safe_buf if label_safe==1 else self.unsafe_buf).push(item)

#         self.global_step += 1

#         return {"t": t, "reward": float(reward), "done": bool(done), "label_safe": int(label_safe),
#                 "h_pi": float(h_pi), "h_qp": float(h_qp), "h_exec": float(h_exe), "alpha": float(alpha)}

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
            return torch.as_tensor(np.stack([getattr(it, attr) for it in items], axis=0),
                                   dtype=torch.float32, device=dev)

        batch = {
            "state_seq":       stack("state_seq"),
            "action_seq":      stack("action_seq"),
            "state_seq_next":  stack("state_seq_next"),
            "action_seq_next": stack("action_seq_next"),
            "u_nom":           stack("u_nom"),
            "u_pi":            stack("u_pi"),
            "u_qp":            stack("u_qp"),
            "u_exec":          stack("u_exec"),
            "y":               torch.as_tensor(np.array([it.label_safe for it in items], dtype=np.float32), device=dev),
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

        # Critic at u_pred
        h_pred = h_score(self.model, st, ac, u_pred)

        # Critic losses
        L_bce = nn.functional.binary_cross_entropy_with_logits(h_pred, y)
        L_margin = ((1-y) * torch.relu(self.cfg.margin + h_pred) + (y) * torch.relu(self.cfg.margin - h_pred)).mean()

        # Gradient penalty
        u_pred_req = u_pred.detach().requires_grad_(True)
        h_tmp = h_score(self.model, st, ac, u_pred_req)
        g = torch.autograd.grad(h_tmp.sum(), u_pred_req, retain_graph=False, create_graph=False)[0]
        L_grad = (g*g).sum(dim=-1).mean()

        # QP teacher on the batch (anchor nominal)
        u_qp_batch = qp_teacher_anchor_nominal(self.model, st, ac, u_nom,
                                               margin=self.cfg.margin, eps=self.cfg.grad_eps, rho=self.cfg.qp_soft_rho)

        # Alpha
        alpha = torch.tensor(self.alpha(), device=dev)

        # Policy losses
        L_teacher = ((u_pred - u_qp_batch)**2).sum(dim=-1) * (1.0 - y)
        L_safe    = ((u_pred - u_nom    )**2).sum(dim=-1) * y
        u_mix_det = ((1.0 - alpha)*u_qp_batch + alpha*u_pred.detach())
        L_exec    = ((u_pred - u_mix_det)**2).sum(dim=-1)
        L_pi = (1.0 - alpha) * L_teacher.mean() + self.cfg.w_pi_safe * L_safe.mean() + self.cfg.w_exec * L_exec.mean()

        # One-step discrete CBF trend (safe labels only)
        h_k   = h_score(self.model, st,  ac,  u_ex)
        h_kp1 = h_score(self.model, stn, acn, u_ex)
        hdot  = (h_kp1 - h_k) / max(self.cfg.dt, 1e-6)
        resid = self.cfg.hdot_lambda * h_k + hdot
        L_hdot = torch.relu(-resid) * y
        L_hdot = L_hdot.mean()

        loss = self.cfg.w_bce*L_bce + self.cfg.w_margin*L_margin + self.cfg.w_grad*L_grad + L_pi + self.cfg.w_hdot*L_hdot

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "L_bce": float(L_bce.item()),
            "L_margin": float(L_margin.item()),
            "L_grad": float(L_grad.item()),
            "L_pi": float(L_pi.item()),
            "L_hdot": float(L_hdot.item()),
        }


#     # ---- sampling / training ----
#     def sample_minibatch(self, batch_size:int) -> Dict[str, torch.Tensor]:
#         n_unsafe = min(len(self.unsafe_buf), batch_size//2)
#         n_safe   = batch_size - n_unsafe
#         items: List[ReplayItem] = []
#         if n_safe>0 and len(self.safe_buf)>0:   items += self.safe_buf.sample(n_safe)
#         if n_unsafe>0 and len(self.unsafe_buf)>0: items += self.unsafe_buf.sample(n_unsafe)
#         if len(items)==0: items = self.primary.sample(batch_size)
#         if len(items)==0: return {}

#         dev = self.device

#         def stack(attr): return torch.as_tensor(np.stack([getattr(it, attr) for it in items], axis=0),
#                                                dtype=torch.float32, device=dev)

#         batch = {
#             "state_seq":       stack("state_seq"),
#             "action_seq":      stack("action_seq"),
#             "state_seq_next":  stack("state_seq_next"),
#             "action_seq_next": stack("action_seq_next"),
#             "u_nom":           stack("u_nom"),
#             "u_pi":            stack("u_pi"),
#             "u_qp":            stack("u_qp"),
#             "u_exec":          stack("u_exec"),
#             "y":               torch.as_tensor(np.array([it.label_safe for it in items], dtype=np.float32), device=dev),
#         }
#         return batch

#     def train_step(self) -> Dict[str, float]:
#         batch = self.sample_minibatch(self.cfg.batch_size)
#         if not batch:
#             return {"loss": 0.0}

#         st   = batch["state_seq"]
#         ac   = batch["action_seq"]
#         stn  = batch["state_seq_next"]
#         acn  = batch["action_seq_next"]
#         u_nom= batch["u_nom"]
#         u_pi = batch["u_pi"]
#         u_qp = batch["u_qp"]
#         u_ex = batch["u_exec"]
#         y    = batch["y"]
#         dev  = self.device

#         # Forward backbone
#         out, _ = self.model.pact({"state": st, "action": ac})
#         last_state = out[:,0::2,:][:,-1,:]

#         # Policy current prediction
#         delta = self.model.policy(last_state)
#         u_pred = torch.clamp(u_nom + delta,
#                              min=torch.as_tensor(self.act_low,  device=dev),
#                              max=torch.as_tensor(self.act_high, device=dev))

#         # Critic at u_pred
#         h_pred = h_score(self.model, st, ac, u_pred)

#         # Critic losses
#         L_bce = nn.functional.binary_cross_entropy_with_logits(h_pred, y)
#         L_margin = ((1-y) * torch.relu(self.cfg.margin + h_pred) + (y) * torch.relu(self.cfg.margin - h_pred)).mean()

#         # grad penalty for smoother rectifier
#         u_pred_req = u_pred.detach().requires_grad_(True)
#         h_tmp = h_score(self.model, st, ac, u_pred_req)
#         g = torch.autograd.grad(h_tmp.sum(), u_pred_req, retain_graph=False, create_graph=False)[0]
#         L_grad = (g*g).sum(dim=-1).mean()

#         # QP teacher on the batch (anchor nominal)
#         u_qp_batch = qp_teacher_anchor_nominal(self.model, st, ac, u_nom, margin=self.cfg.margin, eps=self.cfg.grad_eps)

#         # Alpha
#         alpha = torch.tensor(self.alpha(), device=dev)

#         # Policy losses
#         #  - imitate QP on unsafe (weighted by 1-alpha)
#         #  - stay near nominal on safe
#         #  - consistency with executed mix (stabilizes training distribution)
#         L_teacher = ((u_pred - u_qp_batch)**2).sum(dim=-1) * (1.0 - y)
#         L_safe    = ((u_pred - u_nom    )**2).sum(dim=-1) * y
#         u_mix_det = ((1.0 - alpha)*u_qp_batch + alpha*u_pred.detach())
#         L_exec    = ((u_pred - u_mix_det)**2).sum(dim=-1)
#         L_pi = (1.0 - alpha) * L_teacher.mean() + self.cfg.w_pi_safe * L_safe.mean() + self.cfg.w_exec * L_exec.mean()

#         # hdot finite-difference trend (safe states):  dot{h} + lambda*h >= 0
#         h_k   = h_score(self.model, st,  ac,  u_ex)  # at time t, executed action
#         h_kp1 = h_score(self.model, stn, acn, u_ex)  # evaluate at t+1 repeating u_ex (cheap proxy)
#         hdot  = (h_kp1 - h_k) / max(self.cfg.dt, 1e-6)
#         resid = self.cfg.hdot_lambda * h_k + hdot
#         L_hdot = torch.relu(-resid) * y    # apply only to safe labels
#         L_hdot = L_hdot.mean()

#         loss = self.cfg.w_bce*L_bce + self.cfg.w_margin*L_margin + self.cfg.w_grad*L_grad + L_pi + self.cfg.w_hdot*L_hdot

#         self.opt.zero_grad(set_to_none=True)
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
#         self.opt.step()

#         return {
#             "loss": float(loss.item()),
#             "L_bce": float(L_bce.item()),
#             "L_margin": float(L_margin.item()),
#             "L_grad": float(L_grad.item()),
#             "L_pi": float(L_pi.item()),
#             "L_hdot": float(L_hdot.item()),
#         }
    def run_episode(self, max_steps: Optional[int]=None) -> Dict[str, Any]:
        max_steps = max_steps or self.cfg.max_step
        obs = self.env.reset(seed=self.cfg.seed)
        self.ctx.init(obs["full_obs"].astype(np.float32), np.zeros((2,), dtype=np.float32))

        ep_reward = 0.0
        logs = []
        for t in range(max_steps):
            step_out = self.step_once(t)
            ep_reward += float(step_out["reward"])
            train_out = {"loss": 0.0}
            if self.cfg.update_every > 0 and (t % self.cfg.update_every == 0):
                train_out = self.train_step()
            logs.append({**step_out, **train_out})
            if self.cfg.verbose and (t % 20 == 0):
                msg = (f"[t={t:03d}] r={step_out['reward']:.3f} "
                       f"h_pi={step_out['h_pi']:.3f} h_qp={step_out['h_qp']:.3f} h_exec={step_out['h_exec']:.3f} "
                       f"alpha={step_out['alpha']:.2f} "
                       f"label={step_out['label_safe']} loss={train_out.get('loss',0.0):.4f}")
                if "cbf_score" in step_out:
                    msg += f" cbf_min={step_out.get('cbf_min',0.0):.3f} cbf_score={step_out['cbf_score']:.3f}"
                print(msg)
            if step_out["done"]:
                break

        return {"ep_reward": ep_reward, "steps": t+1, "logs": logs}


    # def run_episode(self, max_steps: Optional[int]=None) -> Dict[str, Any]:
#         max_steps = max_steps or self.cfg.max_step
#         obs = self.env.reset(seed=self.cfg.seed)
#         self.ctx.init(obs["full_obs"].astype(np.float32), np.zeros((2,), dtype=np.float32))

#         ep_reward = 0.0
#         logs = []
#         for t in range(max_steps):
#             step_out = self.step_once(t)
#             ep_reward += float(step_out["reward"])

#             train_out = {"loss": 0.0}
#             if self.cfg.update_every > 0 and (t % self.cfg.update_every == 0):
#                 train_out = self.train_step()

#             logs.append({**step_out, **train_out})
#             if self.cfg.verbose and (t % 20 == 0):
#                 print(f"[t={t:03d}] r={step_out['reward']:.3f} "
#                       f"h_pi={step_out['h_pi']:.3f} h_qp={step_out['h_qp']:.3f} h_exec={step_out['h_exec']:.3f} "
#                       f"alpha={step_out['alpha']:.2f} "
#                       f"label={step_out['label_safe']} loss={train_out.get('loss',0.0):.4f}")
#             if step_out["done"]:
#                 break

#         return {"ep_reward": ep_reward, "steps": t+1, "logs": logs}


# # -----------------------------
# # Convenience API
# # -----------------------------
# import os
# from typing import Optional, Dict, Any
def run_online_episode(cfg: Optional[OnlineConfig]=None) -> Dict[str, Any]:
    cfg = cfg or OnlineConfig()
    os.environ["PACT_DEVICE"] = cfg.device
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
    print("Config:", cfg)

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
        print(f"Ep {episode}: Steps={ep_out['steps']}, Reward={ep_out['ep_reward']:.2f}")


if __name__ == "__main__":
    main()


# """
# PACT Online Agent: End-to-End Online Training Framework (MPC-like CBF Look-Ahead Edition)

# This script implements an online training loop for a transformer-based agent (PACT)
# that learns to navigate safely in a 2D environment with obstacles. The agent uses:
# - A transformer policy network to propose actions based on historical context
# - A critic network (barrier function) to evaluate action safety
# - A QP-based safety teacher to provide safe action corrections
# - An MPC-like look-ahead using the black-box simulator to enforce discrete-time CBF trends
# - An alpha-blending mechanism to gradually shift from teacher to learned policy

# Key Features:
# - Model-free action selection (no explicit dynamics model required)
# - Online updates (learns while acting in the environment)
# - Safety-critical design (Control Barrier Functions, discrete-time)
# - Optional slack in the QP teacher for robustness
# - Short-horizon CBF look-ahead (MPC-like) for adaptive, predictive safety
# - Replay buffer for stable training
# """

# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, Any, List
# import os
# import numpy as np
# import torch
# import torch.nn as nn

# from gcbfplus.env.base import RolloutResult  # type: ignore

# # ---- Project Imports ----
# # Import the PACT transformer model (policy + critic)
# from src.models.modules.a import PACTPolicyCritic  # type: ignore

# # Import the environment: DoubleIntegrator (physics) + SingleAgentDIEnv (wrapper)
# from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv  # type: ignore
# from gcbfplus.env.double_integrator import DoubleIntegrator  # type: ignore


# # =============================
# # Configuration
# # =============================
# @dataclass
# class OnlineConfig:
#     """
#     Configuration dataclass for the online training framework.
#     Contains all hyperparameters for the model, environment, and training process.
#     """
    
#     # === Transformer Model Architecture ===
#     ctx_tokens: int = 16      # Total number of tokens in context window (state + action pairs)
#     n_embd: int = 128         # Embedding dimension for transformer
#     n_layer: int = 4          # Number of transformer layers
#     n_head: int = 8           # Number of attention heads
#     device: str = "cpu"       # "cpu" or "cuda" torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # === Environment Parameters ===
#     area_size: float = 6.0    # Size of the square environment (from -area_size to +area_size) #need to check in the gcbf double integrated environment
#     max_step: int = 256       # Maximum steps per episode
#     dt: float = 0.03          # Timestep duration (seconds)
#     seed: int = 0             # Random seed

#     # === Safety Parameters ===
#     margin: float = 0.15            # gamma, Safety margin for the barrier function (h >= margin)
#     margin_boost: float = 0.3       # Extra margin used when verification fails
#     grad_eps: float = 1e-8          # Epsilon for numerical stability in gradient calculations
#     action_clip: float = 2.0        # Maximum action magnitude (force limits)

#     # Optional slack for the QP teacher (soft projection)
#     # If rho <= 0, the QP is "hard" (no slack). For robustness try rho in [10, 100].
#     qp_soft_rho: float = 0.0

#     # === Verification Parameters ===
#     verify_horizon: int = 8         # Steps to simulate ahead when verifying an action
#     max_verify_attempts: int = 2    # How many times to retry with stronger safety margins

#     # === MPC-like CBF look-ahead (uses black-box simulator) ===
#     mpc_like_horizon: int = 4           # H steps to peek ahead
#     mpc_like_use_qp_first: bool = True  # First action uses u_qp (safer) else the candidate itself
#     mpc_like_repeat: str = "exec"       # {"exec","pi","qp","nom"} action repeated for steps 2..H
#     mpc_like_softmin_tau: float = 5.0   # >0 soft-min aggregation; 0 => hard min
#     w_cbf_hor_online: float = 0.0       # >0 enables a lightweight online gradient step from horizon CBF loss

#     # === Alpha Blending (Teacher-to-Policy Transition) ===
#     alpha_init: float = 0.05        # Initial alpha (0 = full teacher, 1 = full policy)
#     alpha_final: float = 0.90       # Final alpha after warmup
#     alpha_warmup: int = 20000       # Steps to increase alpha from init to final
#     teacher_anchor: str = "nominal" # Anchor point for QP teacher: "nominal" or "policy"

#     # === Training & Optimization ===
#     lr: float = 3e-4                # Learning rate
#     weight_decay: float = 1e-4      # L2 regularization
#     grad_clip: float = 1.0          # Gradient clipping threshold

#     # === Loss Function Weights ===
#     w_bce: float = 1.0          # Binary cross-entropy loss (critic classification)
#     w_margin: float = 0.5       # Margin loss (enforces confident predictions)
#     w_grad: float = 1e-4        # Gradient penalty (smoothness of barrier function)
#     w_pi_safe: float = 0.05     # Policy loss weight for safe state behavior
#     w_exec: float = 0.10        # Execution consistency loss (stabilizes training distribution)
#     w_hdot: float = 0.10        # Barrier derivative loss (enforces CBF condition)

#     # === CBF Derivative Constraint ===
#     hdot_lambda: float = 1.0    # Lambda in discrete-time CBF: (h_{k+1}-h_k)/dt + lambda*h_k >= 0

#     # === Online Update Schedule ===
#     update_every: int = 1           # How often to perform a training update (every N steps)
#     batch_size: int = 64            # Samples per training batch
#     buffer_capacity: int = 20000    # Replay buffer size

#     # === Debugging ===
#     verbose: bool = True            # Print detailed logs during training


# # =============================
# # Replay Buffer Data Structures
# # =============================
# class ReplayItem:
#     """
#     A single experience item stored in the replay buffer.
#     """
#     __slots__ = (
#         "state_seq", "action_seq",
#         "state_seq_next", "action_seq_next",
#         "u_nom", "u_pi", "u_qp", "u_exec",
#         "alpha", "label_safe", "h_t", "h_tp1"
#     )
    
#     def __init__(self, **kw):
#         for k, v in kw.items():
#             setattr(self, k, v)


# class RingBuffer:
#     """
#     Circular replay buffer with overwrite on full.
#     """
#     def __init__(self, capacity: int):
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
    
#     def sample(self, n: int) -> List[ReplayItem]:
#         n = min(n, len(self.data))
#         if n == 0:
#             return []
#         idx = np.random.choice(len(self.data), size=n, replace=False)
#         return [self.data[i] for i in idx]


# # =============================
# # Rolling Context Window
# # =============================
# class RollingContext:
#     """
#     Maintains the fixed-size historical context window for the transformer.
#     """
#     def __init__(self, T: int, S: int, A: int):
#         self.T = T; self.S = S; self.A = A
#         self._states = None
#         self._actions = None
    
#     def init(self, s0: np.ndarray, a0: Optional[np.ndarray] = None):
#         s0 = np.asarray(s0, dtype=np.float32)
#         if a0 is None:
#             a0 = np.zeros((self.A,), dtype=np.float32)
#         a0 = np.asarray(a0, dtype=np.float32)
#         self._states = np.repeat(s0[None, :], self.T, axis=0)   # (T, S)
#         self._actions = np.repeat(a0[None, :], self.T, axis=0)  # (T, A)
    
#     def as_batch(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
#         s = torch.as_tensor(self._states, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, S)
#         a = torch.as_tensor(self._actions, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, A)
#         return s, a
    
#     def after_step_versions(self, s_next: np.ndarray, a_exec: np.ndarray):
#         ns = np.concatenate([self._states[1:], s_next[None, :]], axis=0)   # (T, S)
#         na = np.concatenate([self._actions[1:], a_exec[None, :]], axis=0)  # (T, A)
#         return ns, na
    
#     def update(self, s_next: np.ndarray, a_exec: np.ndarray):
#         self._states, self._actions = self.after_step_versions(s_next, a_exec)


# # =============================
# # Safety Critic & QP Teacher
# # =============================
# def h_score(model: PACTPolicyCritic,
#             state_seq: torch.Tensor, action_seq: torch.Tensor, a_last: torch.Tensor) -> torch.Tensor:
#     """
#     Evaluate the barrier function (critic) for a given action appended to the history.
#     Returns scalar scores per batch (B,).
#     - Positive h => safe
#     - Negative h => unsafe
#     """
#     B, T, A = action_seq.shape
#     a_last = a_last.view(B, 1, A)
#     action_plus = torch.cat([action_seq[:, 1:, :], a_last], dim=1)  # (B, T, A)
#     out_plus, _ = model.pact({"state": state_seq, "action": action_plus})
#     action_ctx_plus = out_plus[:, 1::2, :]  # (B, T, emb_dim)  (odd tokens = action tokens)
#     return model.critic(action_ctx_plus[:, -1, :]).squeeze(-1)  # (B,)


# def qp_teacher_anchor_nominal(model: PACTPolicyCritic,
#                               state_seq: torch.Tensor,
#                               action_seq: torch.Tensor,
#                               u_nom: torch.Tensor,
#                               margin: float,
#                               eps: float = 1e-8,
#                               rho: float = 0.0) -> torch.Tensor:
#     """
#     QP-based safety teacher (closed-form projection onto a linearized half-space).
#     Hard QP:
#         minimize:   0.5 * ||u - u_nom||^2
#         subject to: h(u0) + g^T(u - u0) >= margin
#     Soft QP (if rho>0): allow slack ξ with quadratic penalty; closed-form denominator += 1/rho.

#     Returns:
#         Safe actions u_qp (B, A), detached from autograd graph.
#     """
#     # Anchor at u0 = u_nom and enable gradients to get g
#     u0 = u_nom.detach().clone().requires_grad_(True)  # (B, A)
#     h_per = h_score(model, state_seq, action_seq, u0)  # (B,)
#     g = torch.autograd.grad(h_per.sum(), u0, retain_graph=False, create_graph=False)[0]  # (B, A)

#     # Half-space g^T u >= b, where b = margin - h(u0) + g^T u0
#     b = (margin - h_per).unsqueeze(-1) + (g * u0).sum(-1, keepdim=True)  # (B, 1)
#     gt_unom = (g * u0).sum(-1, keepdim=True)                              # (B, 1)
#     denom = (g * g).sum(-1, keepdim=True).clamp_min(eps)                  # (B, 1)

#     if rho is None or rho <= 0.0:
#         lam = ((b - gt_unom) / denom).clamp_min(0.0)  # hard
#     else:
#         lam = ((b - gt_unom) / (denom + 1.0/float(rho))).clamp_min(0.0)  # soft

#     u_qp = u0 + lam * g  # (B, A)
#     return u_qp.detach()


# # =============================
# # Online Training Runner
# # =============================
# class OnlineRunner:
#     """
#     Orchestrates the online training loop.
#     """
#     def __init__(self, cfg: OnlineConfig):
#         self.cfg = cfg
#         self.device = torch.device(cfg.device)

#         # === Environment Setup ===
#         base = DoubleIntegrator(num_agents=1, area_size=cfg.area_size, max_step=cfg.max_step, dt=cfg.dt)
#         self.env = SingleAgentDIEnv(base, include_velocity=True, normalize_lidar=True,
#                                     full_obs_keys=("state_goal", "lidar"))
        
#         obs = self.env.reset(seed=cfg.seed)
#         full_obs = np.asarray(obs["full_obs"], dtype=np.float32)
#         S = int(full_obs.shape[0])
#         A = 2  # (fx, fy)

#         # === Model Setup ===
#         self.model = PACTPolicyCritic(state_dim=S, action_dim=A,
#                                       ctx_tokens=cfg.ctx_tokens,
#                                       n_embd=cfg.n_embd, n_layer=cfg.n_layer,
#                                       n_head=cfg.n_head,
#                                       action_input_type="continuous").to(self.device)
        
#         self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

#         # === Replay Buffers ===
#         cap = cfg.buffer_capacity
#         self.primary = RingBuffer(cap)
#         self.safe_buf = RingBuffer(cap)
#         self.unsafe_buf = RingBuffer(cap)

#         # === Rolling Context Window ===
#         T = cfg.ctx_tokens // 2
#         self.ctx = RollingContext(T=T, S=S, A=A)
#         self.ctx.init(s0=full_obs, a0=np.zeros((A,), dtype=np.float32))

#         # Bounds
#         self.act_low  = obs.get("action_low",  np.array([-cfg.action_clip, -cfg.action_clip], dtype=np.float32))
#         self.act_high = obs.get("action_high", np.array([ cfg.action_clip,  cfg.action_clip], dtype=np.float32))

#         self.global_step = 0

#     # ---- helpers ----
#     def alpha(self) -> float:
#         if self.cfg.alpha_warmup <= 0: 
#             return float(self.cfg.alpha_final)
#         r = min(1.0, self.global_step / float(self.cfg.alpha_warmup))
#         return float(self.cfg.alpha_init + r * (self.cfg.alpha_final - self.cfg.alpha_init))

#     def nominal_action(self) -> np.ndarray:
#         if hasattr(self.env, "nominal_action"):
#             u = self.env.nominal_action()
#             return np.asarray(u, dtype=np.float32).reshape(-1)
#         return np.zeros((2,), dtype=np.float32)

#     def verify_horizon(self, a0: np.ndarray, H: int) -> bool:
#         if not hasattr(self.env, "rollout_sequence"):
#             return True
#         cand = np.tile(a0.reshape(1,-1), (H,1)).astype(np.float32)
#         states, rewards, infos = self.env.rollout_sequence(cand, early_stop_on_violation=False)
#         for k in range(len(infos)):
#             inf = infos[k] if isinstance(infos, (list,tuple)) else infos
#             if isinstance(inf, dict) and (inf.get("inside_obstacles") or inf.get("unsafe") or inf.get("collision")):
#                 return False
#         return True

#     # ---- CBF helpers ----
#     def _cbf_residual(self, h_now: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
#         dt = max(self.cfg.dt, 1e-6)
#         lam = self.cfg.hdot_lambda
#         return lam * h_now + (h_next - h_now) / dt  # shape: (B,)

#     @torch.no_grad()
#     def cbf_rollout_score(self,
#                           state_seq: torch.Tensor,
#                           action_seq: torch.Tensor,
#                           u_first: torch.Tensor,
#                           u_pi: torch.Tensor,
#                           u_qp: torch.Tensor,
#                           u_nom: torch.Tensor,
#                           H: int) -> Dict[str, Any]:
#         """
#         MPC-like "peek": simulate H steps open-loop and compute discrete CBF residuals r_i.
#         Returns dict with residual list and an aggregate score (soft-min).
#         """
#         dev = self.device
#         if H <= 0 or not hasattr(self.env, "rollout_sequence"):
#             return {"residuals": [], "score": float("inf"), "h": []}

#         # Select the action to repeat after the first step
#         rep = self.cfg.mpc_like_repeat
#         if rep == "pi":
#             u_rep = u_pi
#         elif rep == "qp":
#             u_rep = u_qp
#         elif rep == "nom":
#             u_rep = u_nom
#         else:  # "exec"
#             u_rep = (1.0 - torch.tensor(self.alpha(), device=dev)) * u_qp + torch.tensor(self.alpha(), device=dev) * u_pi

#         # Build candidate sequence for peek
#         first = (u_qp if self.cfg.mpc_like_use_qp_first else u_first).squeeze(0).cpu().numpy()
#         rep_np = u_rep.squeeze(0).cpu().numpy()
#         U = np.vstack([first] + [rep_np for _ in range(max(0, H-1))]).astype(np.float32)  # (H,A)

#         # Query simulator (does not advance real env)
#         states, _, _ = self.env.rollout_sequence(U, early_stop_on_violation=False)
#         states = np.asarray(states, dtype=np.float32)  # (H,S)

#         # Copy windows
#         win_s = self.ctx._states.copy()
#         win_a = self.ctx._actions.copy()

#         residuals = []
#         hs = []

#         # h_0 under first action
#         h_curr = h_score(self.model, state_seq, action_seq,
#                          torch.as_tensor(U[0][None, :], device=dev, dtype=torch.float32)).item()

#         for i in range(H):
#             s_next = states[i]; a_i = U[i]
#             win_s = np.concatenate([win_s[1:], s_next[None, :]], axis=0)
#             win_a = np.concatenate([win_a[1:], a_i[None, :]],  axis=0)

#             st = torch.as_tensor(win_s[None, ...], dtype=torch.float32, device=dev)
#             ac = torch.as_tensor(win_a[None, ...], dtype=torch.float32, device=dev)
#             a_next = torch.as_tensor(U[i if i < H-1 else i], dtype=torch.float32, device=dev).view(1, -1)

#             h_next = h_score(self.model, st, ac, a_next).item()
#             hs.append(h_curr)
#             r_i = self._cbf_residual(torch.tensor(h_curr, device=dev), torch.tensor(h_next, device=dev)).item()
#             residuals.append(float(r_i))
#             h_curr = h_next

#         # Aggregate residuals using soft-min (higher is safer)
#         if self.cfg.mpc_like_softmin_tau > 0:
#             tau = self.cfg.mpc_like_softmin_tau
#             score = float(-tau * torch.logsumexp(-torch.tensor(residuals, device=dev)/tau, dim=0))
#         else:
#             score = float(min(residuals))

#         return {"residuals": residuals, "score": score, "h": hs}

#     def cbf_rollout_loss(self,
#                          state_seq: torch.Tensor,
#                          action_seq: torch.Tensor,
#                          U: np.ndarray) -> torch.Tensor:
#         """
#         Differentiable ONLINE auxiliary loss: mean ReLU(-r_i) over the peeked horizon.
#         Env provides future states (constants); gradients flow through h(.) wrt model params.
#         """
#         dev = self.device
#         if not hasattr(self.env, "rollout_sequence") or len(U) == 0:
#             return torch.tensor(0.0, device=dev)

#         states, _, _ = self.env.rollout_sequence(U.astype(np.float32), early_stop_on_violation=False)
#         states = np.asarray(states, dtype=np.float32)

#         win_s = self.ctx._states.copy()
#         win_a = self.ctx._actions.copy()

#         a0 = torch.as_tensor(U[0][None, :], dtype=torch.float32, device=dev)
#         h_curr = h_score(self.model, state_seq, action_seq, a0)  # (1,)
#         penalties = []

#         for i in range(len(U)):
#             s_next = states[i]; a_i = U[i]
#             win_s = np.concatenate([win_s[1:], s_next[None, :]], axis=0)
#             win_a = np.concatenate([win_a[1:], a_i[None, :]],  axis=0)

#             st = torch.as_tensor(win_s[None, ...], dtype=torch.float32, device=dev)
#             ac = torch.as_tensor(win_a[None, ...], dtype=torch.float32, device=dev)
#             a_next = torch.as_tensor(U[i if i < len(U)-1 else i][None, :], dtype=torch.float32, device=dev)

#             h_next = h_score(self.model, st, ac, a_next)  # (1,)
#             r_i = self._cbf_residual(h_curr, h_next)      # (1,)
#             penalties.append(torch.relu(-r_i))
#             h_curr = h_next

#         return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=dev)

#     # ---- one control step ----
#     def step_once(self, t:int) -> Dict[str, Any]:
#         cfg = self.cfg; dev = self.device

#         # Windows
#         state_seq, action_seq = self.ctx.as_batch(dev)  # (1,T,S), (1,T,A)

#         # Nominal LQR
#         u_nom_np = self.nominal_action()
#         u_nom = torch.as_tensor(u_nom_np, device=dev).unsqueeze(0)  # (1,2)

#         # Policy proposal
#         with torch.no_grad():
#             out, _ = self.model.pact({"state": state_seq, "action": action_seq})
#             last_state = out[:,0::2,:][:,-1,:]
#             delta = self.model.policy(last_state)  # (1,2)
#             u_pi = torch.clamp(u_nom + delta,
#                                min=torch.as_tensor(self.act_low,  device=dev),
#                                max=torch.as_tensor(self.act_high, device=dev))

#         # QP teacher anchored at nominal (optionally soft)
#         u_qp = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom,
#                                          margin=cfg.margin, eps=cfg.grad_eps, rho=cfg.qp_soft_rho)

#         # Alpha-mixed action
#         alpha = self.alpha()
#         u_exec = (1.0 - alpha)*u_qp + alpha*u_pi
#         u_exec = torch.clamp(u_exec,
#                              min=torch.as_tensor(self.act_low,  device=dev),
#                              max=torch.as_tensor(self.act_high, device=dev))

#         # === MPC-like CBF look-ahead selection ===
#         H = int(cfg.mpc_like_horizon)
#         cbf_diag = {}
#         if H > 0 and hasattr(self.env, "rollout_sequence"):
#             scores = {
#                 "exec": self.cbf_rollout_score(state_seq, action_seq, u_exec, u_pi, u_qp, u_nom, H),
#                 "qp":   self.cbf_rollout_score(state_seq, action_seq, u_qp,   u_pi, u_qp, u_nom, H),
#                 "pi":   self.cbf_rollout_score(state_seq, action_seq, u_pi,   u_pi, u_qp, u_nom, H),
#             }
#             choice = max(scores.keys(), key=lambda k: scores[k]["score"])
#             u_chosen = {"exec": u_exec, "qp": u_qp, "pi": u_pi}[choice]
#             cbf_diag = scores[choice]

#             # If even the best horizon looks unsafe, tighten margin and retry QP
#             if scores[choice]["score"] < 0.0 and cfg.max_verify_attempts > 0:
#                 u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom,
#                                                   margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps, rho=cfg.qp_soft_rho)
#                 score2 = self.cbf_rollout_score(state_seq, action_seq, u_qp2, u_pi, u_qp2, u_nom, H)
#                 if score2["score"] > scores[choice]["score"]:
#                     u_chosen = u_qp2
#                     cbf_diag = score2

#             u_exec = torch.clamp(u_chosen,
#                                  min=torch.as_tensor(self.act_low,  device=dev),
#                                  max=torch.as_tensor(self.act_high, device=dev))

#             # Optional: online auxiliary gradient step to reduce horizon CBF violations
#             if cfg.w_cbf_hor_online > 0.0:
#                 rep = cfg.mpc_like_repeat
#                 if rep == "pi":   u_rep = u_pi
#                 elif rep == "qp": u_rep = u_qp
#                 elif rep == "nom":u_rep = u_nom
#                 else:
#                     u_rep = (1.0 - torch.tensor(alpha, device=dev)) * u_qp + torch.tensor(alpha, device=dev) * u_pi
#                 U = np.vstack([u_exec.squeeze(0).cpu().numpy()] +
#                               [u_rep.squeeze(0).cpu().numpy() for _ in range(max(0, H-1))]).astype(np.float32)
#                 L_cbf_hor = self.cbf_rollout_loss(state_seq, action_seq, U) * float(cfg.w_cbf_hor_online)
#                 if L_cbf_hor.item() > 0.0:
#                     self.opt.zero_grad(set_to_none=True)
#                     L_cbf_hor.backward()
#                     nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
#                     self.opt.step()

#         # === Physics verification & repairs (final guard) ===
#         a_try = u_exec.squeeze(0).cpu().numpy()
#         ok = self.verify_horizon(a_try, cfg.verify_horizon)
#         if not ok and cfg.max_verify_attempts > 0:
#             u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom,
#                                               margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps, rho=cfg.qp_soft_rho)
#             u_exec2 = ((1.0 - alpha)*u_qp2 + alpha*u_pi).clamp(
#                 torch.as_tensor(self.act_low, device=dev),
#                 torch.as_tensor(self.act_high, device=dev)
#             )
#             a_try2 = u_exec2.squeeze(0).cpu().numpy()
#             if self.verify_horizon(a_try2, cfg.verify_horizon):
#                 u_exec = u_exec2
#             else:
#                 if self.verify_horizon(u_qp.squeeze(0).cpu().numpy(), cfg.verify_horizon):
#                     u_exec = u_qp
#                 else:
#                     if self.verify_horizon(u_nom.squeeze(0).cpu().numpy(), cfg.verify_horizon):
#                         u_exec = u_nom

#         # Critic reads
#         with torch.no_grad():
#             h_pi  = h_score(self.model, state_seq, action_seq, u_pi).item()
#             h_qp  = h_score(self.model, state_seq, action_seq, u_qp).item()
#             h_exe = h_score(self.model, state_seq, action_seq, u_exec).item()

#         # Execute in real env
#         a_exec_np = u_exec.squeeze(0).cpu().numpy()
#         next_obs, reward, done, info = self.env.step(a_exec_np)
#         label_safe = 1
#         if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
#             label_safe = 0
#         if done and label_safe == 1:
#             label_safe = 0

#         # Next-window versions (for one-step hdot loss)
#         s_next = np.asarray(next_obs["full_obs"], dtype=np.float32)
#         state_seq_next_np, action_seq_next_np = self.ctx.after_step_versions(s_next, a_exec_np)
#         state_seq_next = torch.as_tensor(state_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)
#         action_seq_next= torch.as_tensor(action_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)

#         with torch.no_grad():
#             h_tp1 = h_score(self.model, state_seq_next, action_seq_next, u_exec).item()

#         # Update context
#         self.ctx.update(s_next, a_exec_np)

#         # Store
#         item = ReplayItem(
#             state_seq=state_seq.squeeze(0).cpu().numpy(),
#             action_seq=action_seq.squeeze(0).cpu().numpy(),
#             state_seq_next=state_seq_next.squeeze(0).cpu().numpy(),
#             action_seq_next=action_seq_next.squeeze(0).cpu().numpy(),
#             u_nom=u_nom.squeeze(0).cpu().numpy(),
#             u_pi=u_pi.squeeze(0).cpu().numpy(),
#             u_qp=u_qp.squeeze(0).cpu().numpy(),
#             u_exec=a_exec_np.astype(np.float32),
#             alpha=float(alpha),
#             label_safe=int(label_safe),
#             h_t=float(h_exe),
#             h_tp1=float(h_tp1),
#         )
#         self.primary.push(item)
#         (self.safe_buf if label_safe==1 else self.unsafe_buf).push(item)

#         self.global_step += 1

#         out = {"t": t, "reward": float(reward), "done": bool(done), "label_safe": int(label_safe),
#                "h_pi": float(h_pi), "h_qp": float(h_qp), "h_exec": float(h_exe), "alpha": float(alpha)}
#         if cbf_diag:
#             out["cbf_min"] = float(min(cbf_diag.get("residuals", [0.0])))
#             out["cbf_score"] = float(cbf_diag.get("score", 0.0))
#         return out

#     # ---- sampling / training ----
#     def sample_minibatch(self, batch_size:int) -> Dict[str, torch.Tensor]:
#         n_unsafe = min(len(self.unsafe_buf), batch_size//2)
#         n_safe   = batch_size - n_unsafe
#         items: List[ReplayItem] = []
#         if n_safe>0 and len(self.safe_buf)>0:   items += self.safe_buf.sample(n_safe)
#         if n_unsafe>0 and len(self.unsafe_buf)>0: items += self.unsafe_buf.sample(n_unsafe)
#         if len(items)==0: items = self.primary.sample(batch_size)
#         if len(items)==0: return {}

#         dev = self.device
#         def stack(attr): 
#             return torch.as_tensor(np.stack([getattr(it, attr) for it in items], axis=0),
#                                    dtype=torch.float32, device=dev)

#         batch = {
#             "state_seq":       stack("state_seq"),
#             "action_seq":      stack("action_seq"),
#             "state_seq_next":  stack("state_seq_next"),
#             "action_seq_next": stack("action_seq_next"),
#             "u_nom":           stack("u_nom"),
#             "u_pi":            stack("u_pi"),
#             "u_qp":            stack("u_qp"),
#             "u_exec":          stack("u_exec"),
#             "y":               torch.as_tensor(np.array([it.label_safe for it in items], dtype=np.float32), device=dev),
#         }
#         return batch

#     def train_step(self) -> Dict[str, float]:
#         batch = self.sample_minibatch(self.cfg.batch_size)
#         if not batch:
#             return {"loss": 0.0}

#         st   = batch["state_seq"]
#         ac   = batch["action_seq"]
#         stn  = batch["state_seq_next"]
#         acn  = batch["action_seq_next"]
#         u_nom= batch["u_nom"]
#         u_pi = batch["u_pi"]
#         u_qp = batch["u_qp"]
#         u_ex = batch["u_exec"]
#         y    = batch["y"]
#         dev  = self.device

#         # Forward backbone
#         out, _ = self.model.pact({"state": st, "action": ac})
#         last_state = out[:,0::2,:][:,-1,:]

#         # Policy current prediction
#         delta = self.model.policy(last_state)
#         u_pred = torch.clamp(u_nom + delta,
#                              min=torch.as_tensor(self.act_low,  device=dev),
#                              max=torch.as_tensor(self.act_high, device=dev))

#         # Critic at u_pred
#         h_pred = h_score(self.model, st, ac, u_pred)

#         # Critic losses
#         L_bce = nn.functional.binary_cross_entropy_with_logits(h_pred, y)
#         L_margin = ((1-y) * torch.relu(self.cfg.margin + h_pred) + (y) * torch.relu(self.cfg.margin - h_pred)).mean()

#         # Gradient penalty
#         u_pred_req = u_pred.detach().requires_grad_(True)
#         h_tmp = h_score(self.model, st, ac, u_pred_req)
#         g = torch.autograd.grad(h_tmp.sum(), u_pred_req, retain_graph=False, create_graph=False)[0]
#         L_grad = (g*g).sum(dim=-1).mean()

#         # QP teacher on the batch (anchor nominal)
#         u_qp_batch = qp_teacher_anchor_nominal(self.model, st, ac, u_nom,
#                                                margin=self.cfg.margin, eps=self.cfg.grad_eps, rho=self.cfg.qp_soft_rho)

#         # Alpha
#         alpha = torch.tensor(self.alpha(), device=dev)

#         # Policy losses
#         L_teacher = ((u_pred - u_qp_batch)**2).sum(dim=-1) * (1.0 - y)
#         L_safe    = ((u_pred - u_nom    )**2).sum(dim=-1) * y
#         u_mix_det = ((1.0 - alpha)*u_qp_batch + alpha*u_pred.detach())
#         L_exec    = ((u_pred - u_mix_det)**2).sum(dim=-1)
#         L_pi = (1.0 - alpha) * L_teacher.mean() + self.cfg.w_pi_safe * L_safe.mean() + self.cfg.w_exec * L_exec.mean()

#         # One-step discrete CBF trend (safe labels only)
#         h_k   = h_score(self.model, st,  ac,  u_ex)
#         h_kp1 = h_score(self.model, stn, acn, u_ex)
#         hdot  = (h_kp1 - h_k) / max(self.cfg.dt, 1e-6)
#         resid = self.cfg.hdot_lambda * h_k + hdot
#         L_hdot = torch.relu(-resid) * y
#         L_hdot = L_hdot.mean()

#         loss = self.cfg.w_bce*L_bce + self.cfg.w_margin*L_margin + self.cfg.w_grad*L_grad + L_pi + self.cfg.w_hdot*L_hdot

#         self.opt.zero_grad(set_to_none=True)
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
#         self.opt.step()

#         return {
#             "loss": float(loss.item()),
#             "L_bce": float(L_bce.item()),
#             "L_margin": float(L_margin.item()),
#             "L_grad": float(L_grad.item()),
#             "L_pi": float(L_pi.item()),
#             "L_hdot": float(L_hdot.item()),
#         }

#     def run_episode(self, max_steps: Optional[int]=None) -> Dict[str, Any]:
#         max_steps = max_steps or self.cfg.max_step
#         obs = self.env.reset(seed=self.cfg.seed)
#         self.ctx.init(obs["full_obs"].astype(np.float32), np.zeros((2,), dtype=np.float32))

#         ep_reward = 0.0
#         logs = []
#         for t in range(max_steps):
#             step_out = self.step_once(t)
#             ep_reward += float(step_out["reward"])
#             train_out = {"loss": 0.0}
#             if self.cfg.update_every > 0 and (t % self.cfg.update_every == 0):
#                 train_out = self.train_step()
#             logs.append({**step_out, **train_out})
#             if self.cfg.verbose and (t % 20 == 0):
#                 msg = (f"[t={t:03d}] r={step_out['reward']:.3f} "
#                        f"h_pi={step_out['h_pi']:.3f} h_qp={step_out['h_qp']:.3f} h_exec={step_out['h_exec']:.3f} "
#                        f"alpha={step_out['alpha']:.2f} "
#                        f"label={step_out['label_safe']} loss={train_out.get('loss',0.0):.4f}")
#                 if "cbf_score" in step_out:
#                     msg += f" cbf_min={step_out.get('cbf_min',0.0):.3f} cbf_score={step_out['cbf_score']:.3f}"
#                 print(msg)
#             if step_out["done"]:
#                 break

#         return {"ep_reward": ep_reward, "steps": t+1, "logs": logs}


# # -----------------------------
# # Convenience API
# # -----------------------------
# def run_online_episode(cfg: Optional[OnlineConfig]=None) -> Dict[str, Any]:
#     cfg = cfg or OnlineConfig()
#     os.environ["PACT_DEVICE"] = cfg.device
#     runner = OnlineRunner(cfg)
#     return runner.run_episode()


# # =============================
# # Main Entry Point
# # =============================
# def main():
#     cfg = OnlineConfig()
#     print("Config:", cfg)
#     runner = OnlineRunner(cfg)
#     for episode in range(10):  # keep small by default
#         ep_out = runner.run_episode()
#         print(f"Ep {episode}: Steps={ep_out['steps']}, Reward={ep_out['ep_reward']:.2f}")


# if __name__ == "__main__":
#     main()


#bacup old

# PACT Online Agent: End-to-End Online Training Framework

# This script implements an online training loop for a transformer-based agent (PACT)
# that learns to navigate safely in a 2D environment with obstacles. The agent uses:
# - A transformer policy network to propose actions based on historical context
# - A critic network (barrier function) to evaluate action safety
# - A QP-based safety teacher to provide safe action corrections
# - An alpha-blending mechanism to gradually shift from teacher to learned policy

# Key Features:
# - Model-free learning (no dynamics model required)
# - Online updates (learns while acting in the environment)
# - Safety-critical design (uses Control Barrier Functions)
# - Replay buffer for stable training
# """

# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, Any, List
# import numpy as np
# import torch
# import torch.nn as nn
# import argparse
# from gcbfplus.env.base import RolloutResult

# device = "cpu"

# # ---- Project Imports ----
# # Import the PACT transformer model (policy + critic)
# from src.models.modules.a import PACTPolicyCritic

# # Import the environment: DoubleIntegrator (physics) + SingleAgentDIEnv (wrapper)
# from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv
# from gcbfplus.env.double_integrator import DoubleIntegrator


# # =============================
# # Configuration
# # =============================
# @dataclass
# class OnlineConfig:
#     """
#     Configuration dataclass for the online training framework.
#     Contains all hyperparameters for the model, environment, and training process.
#     """
    
#     # === Transformer Model Architecture ===
#     ctx_tokens: int = 16      # Total number of tokens in context window (state + action pairs)
#     n_embd: int = 128         # Embedding dimension for transformer
#     n_layer: int = 4          # Number of transformer layers
#     n_head: int = 8           # Number of attention heads
#     device: str = "cpu"       # Device to run on ("cpu" or "cuda"), will change it later, when all integration will be finished
    
#     # === Environment Parameters ===
#     area_size: float = 6.0    # Size of the square environment (from -area_size to +area_size)
#     max_step: int = 256       # Maximum steps per episode
#     dt: float = 0.03          # Timestep duration (seconds)
#     seed: int = 0             # Random seed for reproducibility

#     # === Safety Parameters ===
#     # These control the QP-based safety teacher that corrects unsafe actions
#     margin: float = 0.15            # Safety margin for the barrier function (h >= margin for safe)
#     margin_boost: float = 0.3       # Additional margin used when a verification fails
#     grad_eps: float = 1e-8          # Small epsilon for numerical stability in gradient calculations
#     action_clip: float = 2.0        # Maximum action magnitude (force limits)

#     # === Verification Parameters ===
#     # These control the forward simulation used to verify action safety
#     verify_horizon: int = 8         # Number of steps to simulate ahead when verifying an action
#     max_verify_attempts: int = 2    # How many times to retry with stronger safety margins

#     # === Alpha Blending (Teacher-to-Policy Transition) ===
#     # Controls the gradual shift from relying on the QP teacher to trusting the learned policy
#     alpha_init: float = 0.05        # Initial alpha (0 = full teacher, 1 = full policy)
#     alpha_final: float = 0.90       # Final alpha after warmup
#     alpha_warmup: int = 20000       # Number of steps over which alpha increases from init to final
#     teacher_anchor: str = "nominal" # Anchor point for QP teacher: "nominal" or "policy"

#     # === Training & Optimization ===
#     lr: float = 3e-4                # Learning rate for Adam optimizer
#     weight_decay: float = 1e-4      # L2 regularization weight
#     grad_clip: float = 1.0          # Gradient clipping threshold (prevents exploding gradients)

#     # === Loss Function Weights ===
#     # These control the relative importance of each loss component
#     w_bce: float = 1.0          # Binary cross-entropy loss (critic classification)
#     w_margin: float = 0.5       # Margin loss (enforces confident predictions)
#     w_grad: float = 1e-4        # Gradient penalty (smoothness of barrier function)
#     w_pi_safe: float = 0.05     # Policy loss weight for safe state behavior
#     w_exec: float = 0.10        # Execution consistency loss (stabilizes training distribution)
#     w_hdot: float = 0.10        # Barrier derivative loss (enforces CBF condition)

#     # === CBF Derivative Constraint ===
#     hdot_lambda: float = 1.0    # Lambda parameter for discrete-time CBF: h_dot + lambda*h >= 0

#     # === Online Update Schedule ===
#     update_every: int = 1           # How often to perform a training update (every N steps)
#     batch_size: int = 64            # Number of samples per training batch
#     buffer_capacity: int = 20000    # Maximum size of the replay buffer

#     # === Debugging ===
#     verbose: bool = True            # Whether to print detailed logs during training

# # =============================
# # Replay Buffer Data Structures
# # =============================
# class ReplayItem:
#     """
#     A single experience item stored in the replay buffer.
    
#     Contains all information needed to train the agent from one timestep:
#     - Current and next state/action windows (for the transformer context)
#     - All action variants (nominal, policy, QP, executed)
#     - Safety label and barrier values (for computing losses)
#     """
#     __slots__ = (
#         "state_seq",        # Current state window (T, S) - input to transformer at time t
#         "action_seq",       # Current action window (T, A) - input to transformer at time t
#         "state_seq_next",   # Next state window (T, S) - for computing h(t+1) in CBF constraint
#         "action_seq_next",  # Next action window (T, A) - for computing h(t+1) in CBF constraint
#         "u_nom",            # Nominal action from LQR controller (goal-seeking, no safety)
#         "u_pi",             # Policy action from transformer (learned behavior)
#         "u_qp",             # QP teacher action (safety-corrected)
#         "u_exec",           # Actually executed action (blend of QP and policy)
#         "alpha",            # Blending coefficient used at this timestep
#         "label_safe",       # Ground truth safety label: 1=safe, 0=unsafe (collision)
#         "h_t",              # Barrier value at time t
#         "h_tp1"             # Barrier value at time t+1 (for CBF derivative constraint)
#     )
    
#     def __init__(self, **kw):
#         """Initialize by setting all provided keyword arguments as attributes."""
#         for k, v in kw.items():
#             setattr(self, k, v)


# class RingBuffer:
#     """
#     Circular replay buffer for storing agent experiences.
    
#     Implements a fixed-size buffer that overwrites the oldest data when full.
#     Supports random sampling for training batches.
#     """
    
#     def __init__(self, capacity: int):
#         """
#         Initialize an empty buffer with a given capacity.
        
#         Args:
#             capacity: Maximum number of experiences to store
#         """
#         self.capacity = int(capacity)
#         self.data: List[ReplayItem] = []  # Storage for experiences
#         self.ptr = 0                      # Write pointer for circular overwriting
    
#     def __len__(self):
#         """Return the current number of items in the buffer."""
#         return len(self.data)
    
#     def push(self, item: ReplayItem):
#         """
#         Add a new experience to the buffer.
        
#         If the buffer is full, overwrites the oldest item.
        
#         Args:
#             item: ReplayItem to store
#         """
#         if len(self.data) < self.capacity:
#             # Buffer not full yet - just append
#             self.data.append(item)
#         else:
#             # Buffer full - overwrite oldest item and advance pointer
#             self.data[self.ptr] = item
#             self.ptr = (self.ptr + 1) % self.capacity
    
#     def sample(self, n: int) -> List[ReplayItem]:
#         """
#         Randomly sample n experiences from the buffer.
        
#         Args:
#             n: Number of samples to draw
            
#         Returns:
#             List of n ReplayItems (or fewer if buffer is smaller than n)
#         """
#         n = min(n, len(self.data))
#         if n == 0:
#             return []
#         # Sample without replacement
#         idx = np.random.choice(len(self.data), size=n, replace=False)
#         return [self.data[i] for i in idx]


# # =============================
# # Rolling Context Window
# # =============================
# class RollingContext:
#     """
#     Manages the fixed-size historical context window for the transformer.
    
#     The transformer needs to see the last T timesteps of (state, action) pairs.
#     This class maintains a sliding window that:
#     - Starts by repeating the initial state/action
#     - Shifts left and adds new data as the agent acts
#     - Provides efficient conversion to PyTorch tensors for model input
    
#         States:  [s0, s0, s0]  ← Initial state repeated 3 times
#         Actions: [a0, a0, a0]  ← Zero action repeated 3 times
#         Step 1 (After first action):
#         States:  [s0, s0, s1]  ← Shifted left, added new state s1
#         Actions: [a0, a0, a1]  ← Shifted left, added executed action a1
#         Step 2 (After second action):
#         States:  [s0, s1, s2]  ← Shifted left, added new state s2
#         Actions: [a0, a1, a2]  ← Shifted left, added executed action a2
#         Step 3 (After third action):
#         States:  [s1, s2, s3]  ← s0 "falls off" the left end
#         Actions: [a1, a2, a3]  ← a0 "falls off" the left end
#     """
    
#     def __init__(self, T: int, S: int, A: int):
#         """
#         Initialize an empty context window.
        
#         Args:
#             T: Context length (number of timesteps to remember)
#             S: State dimension
#             A: Action dimension
#         """
#         self.T = T  # Window length
#         self.S = S  # State dimension
#         self.A = A  # Action dimension
#         self._states = None   # Will be (T, S) numpy array
#         self._actions = None  # Will be (T, A) numpy array
    
#     def init(self, s0: np.ndarray, a0: Optional[np.ndarray] = None):
#         """
#         Initialize the window by repeating the initial state and action T times.
        
#         This is done at the start of an episode to create a valid input for the transformer.
        
#         Args:
#             s0: Initial state (S,)
#             a0: Initial action (A,), defaults to zero if not provided
#         """
#         s0 = np.asarray(s0, dtype=np.float32)
#         if a0 is None:
#             a0 = np.zeros((self.A,), dtype=np.float32)
#         a0 = np.asarray(a0, dtype=np.float32)
#         # Repeat the initial state and action T times to fill the window
#         self._states = np.repeat(s0[None, :], self.T, axis=0)   # (T, S)
#         self._actions = np.repeat(a0[None, :], self.T, axis=0)  # (T, A)
    
#     def as_batch(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Convert the current window to PyTorch tensors with batch dimension.
        
#         Args:
#             device: PyTorch device to place tensors on
            
#         Returns:
#             Tuple of (state_seq, action_seq) both with shape (1, T, D)
#         """
#         s = torch.as_tensor(self._states, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, S)
#         a = torch.as_tensor(self._actions, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, A)
#         return s, a
    
#     def after_step_versions(self, s_next: np.ndarray, a_exec: np.ndarray):
#         """
#         Compute what the window would look like after taking a step.
        
#         This is used for computing h(t+1) in the CBF derivative constraint.
#         Does NOT modify the internal state.
        
#         Args:
#             s_next: Next state observed after taking action (S,)
#             a_exec: Action that was executed (A,)
            
#         Returns:
#             Tuple of (next_states, next_actions) both with shape (T, D)
#         """
#         # Shift the window left by 1 and append the new data
#         ns = np.concatenate([self._states[1:], s_next[None, :]], axis=0)   # (T, S)
#         na = np.concatenate([self._actions[1:], a_exec[None, :]], axis=0)  # (T, A)
#         return ns, na
    
#     def update(self, s_next: np.ndarray, a_exec: np.ndarray):
#         """
#         Update the internal window by shifting left and adding new data.
        
#         This is called after each environment step to maintain the rolling context.
        
#         Args:
#             s_next: Next state observed after taking action (S,)
#             a_exec: Action that was executed (A,)
#         """
#         self._states, self._actions = self.after_step_versions(s_next, a_exec)


# # =============================
# # Safety Critic & QP Teacher
# # =============================
# def h_score(model: PACTPolicyCritic,
#             state_seq: torch.Tensor, action_seq: torch.Tensor, a_last: torch.Tensor) -> torch.Tensor:
#     """
#     Evaluate the barrier function (critic) for a given action.
    
#     The critic looks at a historical context (state_seq, action_seq) and predicts
#     the safety of adding action a_last to the end of the sequence.
    
#     Process:
#     1. Shift the action sequence left by 1 timestep
#     2. Append a_last to the end
#     3. Pass through the transformer to get embeddings
#     4. Extract the final action token embedding
#     5. Pass through the critic head to get a scalar safety score
    
#     Args:
#         model: PACTPolicyCritic (transformer + critic head)
#         state_seq: Historical states (B, T, S)
#         action_seq: Historical actions (B, T, A)
#         a_last: Action to evaluate (B, A)
    
#     Returns:
#         Safety scores h(s, a) for each sample in the batch (B,)
#         - h > 0 means safe
#         - h < 0 means unsafe
#     """
#     B, T, A = action_seq.shape
#     a_last = a_last.view(B, 1, A)
#     # Create new action sequence: [action_seq[1:], a_last]
#     action_plus = torch.cat([action_seq[:, 1:, :], a_last], dim=1)  # (B, T, A)
    
#     # Forward pass through transformer
#     out_plus, _ = model.pact({"state": state_seq, "action": action_plus})
    
#     # Extract action tokens (transformer outputs tokens for both state and action)
#     # Convention: out[:, 0::2] = state tokens, out[:, 1::2] = action tokens
#     action_ctx_plus = out_plus[:, 1::2, :]  # (B, T, emb_dim)
    
#     # Pass the final action token through the critic head
#     return model.critic(action_ctx_plus[:, -1, :]).squeeze(-1)  # (B,)


# def qp_teacher_anchor_nominal(model: PACTPolicyCritic,
#                               state_seq: torch.Tensor,
#                               action_seq: torch.Tensor,
#                               u_nom: torch.Tensor,
#                               margin: float,
#                               eps: float = 1e-8) -> torch.Tensor:
#     """
#     Quadratic Programming (QP) based safety teacher.
    
#     Solves the optimization problem:
#         minimize:   0.5 * ||u - u_nom||^2
#         subject to: h(u0) + g^T(u - u0) >= margin
    
#     Where:
#     - u_nom is the nominal (goal-seeking) action
#     - h is the barrier function (critic)
#     - g is the gradient of h with respect to the action
#     - margin is the safety threshold
    
#     This finds the action closest to u_nom that satisfies the safety constraint.
#     The constraint is linearized around u0 = u_nom for computational efficiency.
    
#     Mathematical derivation:
#     1. The Lagrangian is: L = 0.5||u - u_nom||^2 - λ(h(u0) + g^T(u - u0) - margin)
#     2. Setting ∇_u L = 0 gives: u - u_nom - λg = 0, so u = u_nom + λg
#     3. The KKT complementarity condition determines λ
#     4. If the constraint is inactive (already safe), λ=0 and u = u_nom
#     5. If active, λ is solved from the constraint: g^T(u_nom + λg) = margin - h(u0) + g^T u0
    
#     example : 
    
#     Args:
#         model: PACTPolicyCritic network
#         state_seq: Historical states (B, T, S)
#         action_seq: Historical actions (B, T, A)
#         u_nom: Nominal actions to anchor the QP (B, A)
#         margin: Safety margin (h must be >= margin)
#         eps: Small constant for numerical stability
    
#     Returns:
#         Safe actions u_qp (B, A) that are closest to u_nom while respecting safety

#     Example:
#     (u_nom=[0,0])
#     critic says (h(u_nom)=-0.2) (unsafe)
#     desired margin (=0.1) → we need to gain (0.1 - (-0.2)=0.3) of safety
#     gradient (g=[2,,1])

#     compute pieces:

#     |g|^2 = 2^2 + 1^2 = 4 + 1 = 5 
#     lambda = (0.3) / 5 = 0.06   (since gap is (0.3))

#     new action: ( u_qp = [0,0] + 0.06*[2,1] = [0.12,0.06] )

#     sanity check (does the linearized constraint hit the margin?):
#     (g^T u_qp = 2* 0.12 + 1* 0.06 = 0.24 + 0.06 = 0.30)

#     linearized score at (u_qp): (h(u_nom) + g^T(u_{qp}-u_{nom}) = -0.2 + 0.30 = 0.10 = {margin}) 

#     already safe case: if (h(u_{nom})=0.2) and margin (=0.1) → gap (= -0.1) → (lambda=0) → (u_{qp}=u_{nom}).
#     """
#     # Evaluate h and its gradient at u0 = u_nom
#     u0 = u_nom.detach().clone().requires_grad_(True)  # (B, A)
    
#     # Compute h(u0) for each sample in the batch
#     h_per = h_score(model, state_seq, action_seq, u0)  # (B,)
#     h_sum = h_per.sum()  # Sum for backprop
    
#     # Compute gradient g = ∂h/∂u at u = u0
#     g = torch.autograd.grad(h_sum, u0, retain_graph=False, create_graph=False)[0]  # (B, A)

#     # Define the safety constraint as a half-space: g^T u >= b
#     # where b = margin - h(u0) + g^T u0
#     b = (margin - h_per).unsqueeze(-1) + (g * u0).sum(-1, keepdim=True)  # (B, 1)
    
#     # Denominator for projection: ||g||^2
#     denom = (g * g).sum(-1, keepdim=True).clamp_min(eps)  # (B, 1)
    
#     # Check if u_nom violates the constraint
#     gt_unom = (g * u0).sum(-1, keepdim=True)  # (B, 1) = g^T u_nom
    
#     # Lagrange multiplier λ (KKT condition)
#     # If λ > 0, the constraint is active and we project u_nom onto the boundary
#     # If λ ≤ 0, u_nom already satisfies the constraint, so use it as-is
#     lam = ((b - gt_unom) / denom).clamp_min(0.0)  # (B, 1)
    
#     # Final safe action: u_qp = u_nom + λ * g
#     u_qp = u0 + lam * g  # (B, A)
    
#     return u_qp.detach()  # Detach to prevent gradients flowing back through QP


# # =============================
# # Online Training Runner
# # =============================
# class OnlineRunner:
#     """
#     Main class that orchestrates the online training loop.
    
#     Responsibilities:
#     1. Initialize and manage the environment
#     2. Initialize and manage the transformer model
#     3. Maintain replay buffers for training
#     4. Execute the agent-environment interaction loop
#     5. Trigger training updates at regular intervals
#     6. Blend actions from the policy and safety teacher
    
#     The training is "online" because the agent learns continuously as it interacts
#     with the environment, rather than collecting data first and then training.
#     """
    
#     def __init__(self, cfg: OnlineConfig):
#         """
#         Initialize the online training runner.
        
#         Args:
#             cfg: Configuration object with all hyperparameters
#         """
#         self.cfg = cfg
#         self.device = torch.device(cfg.device)

#         # === Environment Setup ===
#         # Create the physics simulator (DoubleIntegrator)
#         base = DoubleIntegrator(
#             num_agents=1, 
#             area_size=cfg.area_size, 
#             max_step=cfg.max_step, 
#             dt=cfg.dt
#         )
#         # Wrap with SingleAgentDIEnv to get a clean Gym-like interface
#         self.env = SingleAgentDIEnv(
#             base, 
#             include_velocity=True,           # Include velocity in state observation
#             normalize_lidar=True,            # Normalize LiDAR readings to [0, 1]
#             full_obs_keys=("state_goal", "lidar")  # Components to include in observation
#         )
        
#         # Reset environment to get initial observation and determine dimensions
#         obs = self.env.reset(seed=cfg.seed)
#         full_obs = np.asarray(obs["full_obs"], dtype=np.float32)
#         S = int(full_obs.shape[0])  # State dimension
#         A = 2                        # Action dimension (fx, fy)

#         # === Model Setup ===
#         # Create the PACT transformer model (policy + critic)
#         self.model = PACTPolicyCritic(
#             state_dim=S, 
#             action_dim=A, 
#             ctx_tokens=cfg.ctx_tokens,  # Total context length (state + action pairs)
#             n_embd=cfg.n_embd,          # Embedding dimension
#             n_layer=cfg.n_layer,        # Number of transformer layers
#             n_head=cfg.n_head,          # Number of attention heads
#             action_input_type="continuous"  # Continuous action space
#         ).to(self.device)
        
#         # Adam optimizer with weight decay (L2 regularization)
#         self.opt = torch.optim.AdamW(
#             self.model.parameters(), 
#             lr=cfg.lr, 
#             weight_decay=cfg.weight_decay
#         )

#         # === Replay Buffers ===
#         # We maintain 3 buffers to ensure balanced sampling of safe/unsafe experiences
#         cap = cfg.buffer_capacity
#         self.primary = RingBuffer(cap)    # All experiences
#         self.safe_buf = RingBuffer(cap)   # Only safe experiences
#         self.unsafe_buf = RingBuffer(cap) # Only unsafe experiences

#         # === Rolling Context Window ===
#         # The transformer needs to see T timesteps of history
#         # Since each timestep has 2 tokens (state + action), ctx_tokens = 2*T
#         T = cfg.ctx_tokens // 2
#         self.ctx = RollingContext(T=T, S=S, A=A)
#         # Initialize with the starting state and zero action
#         self.ctx.init(s0=full_obs, a0=np.zeros((A,), dtype=np.float32))

#         # Bounds
#         self.act_low  = obs.get("action_low",  np.array([-cfg.action_clip, -cfg.action_clip], dtype=np.float32))
#         self.act_high = obs.get("action_high", np.array([ cfg.action_clip,  cfg.action_clip], dtype=np.float32))

#         self.global_step = 0

#     # ---- helpers ----
#     def alpha(self) -> float:
#         if self.cfg.alpha_warmup <= 0: return float(self.cfg.alpha_final)
#         r = min(1.0, self.global_step / float(self.cfg.alpha_warmup))
#         return float(self.cfg.alpha_init + r * (self.cfg.alpha_final - self.cfg.alpha_init))

#     def nominal_action(self) -> np.ndarray:
#         if hasattr(self.env, "nominal_action"):
#             u = self.env.nominal_action()
#             return np.asarray(u, dtype=np.float32).reshape(-1)
#         return np.zeros((2,), dtype=np.float32)

#     def verify_horizon(self, a0: np.ndarray, H: int) -> bool:
#         if not hasattr(self.env, "rollout_sequence"):
#             return True
#         cand = np.tile(a0.reshape(1,-1), (H,1)).astype(np.float32)
#         states, rewards, infos = self.env.rollout_sequence(cand, early_stop_on_violation=False)
#         for k in range(len(infos)):
#             inf = infos[k] if isinstance(infos, (list,tuple)) else infos
#             if isinstance(inf, dict) and (inf.get("inside_obstacles") or inf.get("unsafe") or inf.get("collision")):
#                 return False
#         return True

#     # ---- one control step ----
#     def step_once(self, t:int) -> Dict[str, Any]:
#         cfg = self.cfg; dev = self.device

#         # Windows
#         state_seq, action_seq = self.ctx.as_batch(dev)  # (1,T,S), (1,T,A)

#         # Nominal LQR
#         u_nom_np = self.nominal_action()
#         u_nom = torch.as_tensor(u_nom_np, device=dev).unsqueeze(0)  # (1,2)

#         # Policy proposal
#         with torch.no_grad():
#             out, _ = self.model.pact({"state": state_seq, "action": action_seq})
#             last_state = out[:,0::2,:][:,-1,:]
#             delta = self.model.policy(last_state)  # (1,2)
#             u_pi = torch.clamp(u_nom + delta,
#                                min=torch.as_tensor(self.act_low,  device=dev),
#                                max=torch.as_tensor(self.act_high, device=dev))

#         # QP teacher anchored at nominal
#         u_qp = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom, margin=cfg.margin, eps=cfg.grad_eps)

#         # Alpha-mixed action
#         alpha = self.alpha()
#         u_exec = (1.0 - alpha)*u_qp + alpha*u_pi
#         u_exec = torch.clamp(u_exec,
#                              min=torch.as_tensor(self.act_low,  device=dev),
#                              max=torch.as_tensor(self.act_high, device=dev))

#         # Horizon verification & repairs
#         a_try = u_exec.squeeze(0).cpu().numpy()
#         ok = self.verify_horizon(a_try, cfg.verify_horizon)
#         if not ok and cfg.max_verify_attempts > 0:
#             # stronger margin
#             u_qp2 = qp_teacher_anchor_nominal(self.model, state_seq, action_seq, u_nom, margin=cfg.margin + cfg.margin_boost, eps=cfg.grad_eps)
#             u_exec2 = ((1.0 - alpha)*u_qp2 + alpha*u_pi).clamp(
#                 torch.as_tensor(self.act_low, device=dev),
#                 torch.as_tensor(self.act_high, device=dev)
#             )
#             a_try2 = u_exec2.squeeze(0).cpu().numpy()
#             if self.verify_horizon(a_try2, cfg.verify_horizon):
#                 u_exec = u_exec2
#             else:
#                 # fall back to pure QP if that passes
#                 if self.verify_horizon(u_qp.squeeze(0).cpu().numpy(), cfg.verify_horizon):
#                     u_exec = u_qp
#                 else:
#                     # last resort: nominal if preview says ok
#                     if self.verify_horizon(u_nom.squeeze(0).cpu().numpy(), cfg.verify_horizon):
#                         u_exec = u_nom

#         # Critic reads
#         with torch.no_grad():
#             h_pi  = h_score(self.model, state_seq, action_seq, u_pi).item()
#             h_qp  = h_score(self.model, state_seq, action_seq, u_qp).item()
#             h_exe = h_score(self.model, state_seq, action_seq, u_exec).item()

#         # Execute in real env
#         a_exec_np = u_exec.squeeze(0).cpu().numpy()
#         next_obs, reward, done, info = self.env.step(a_exec_np)
#         label_safe = 1
#         if isinstance(info, dict) and (info.get("inside_obstacles") or info.get("unsafe") or info.get("collision")):
#             label_safe = 0
#         if done and label_safe == 1:
#             # treat terminal as unsafe unless explicitly marked safe
#             label_safe = 0

#         # Next-window versions (for hdot)
#         s_next = np.asarray(next_obs["full_obs"], dtype=np.float32)
#         state_seq_next_np, action_seq_next_np = self.ctx.after_step_versions(s_next, a_exec_np)
#         state_seq_next = torch.as_tensor(state_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)
#         action_seq_next= torch.as_tensor(action_seq_next_np, dtype=torch.float32, device=dev).unsqueeze(0)

#         with torch.no_grad():
#             h_tp1 = h_score(self.model, state_seq_next, action_seq_next, u_exec).item()

#         # Update context
#         self.ctx.update(s_next, a_exec_np)

#         # Store
#         item = ReplayItem(
#             state_seq=state_seq.squeeze(0).cpu().numpy(),
#             action_seq=action_seq.squeeze(0).cpu().numpy(),
#             state_seq_next=state_seq_next.squeeze(0).cpu().numpy(),
#             action_seq_next=action_seq_next.squeeze(0).cpu().numpy(),
#             u_nom=u_nom.squeeze(0).cpu().numpy(),
#             u_pi=u_pi.squeeze(0).cpu().numpy(),
#             u_qp=u_qp.squeeze(0).cpu().numpy(),
#             u_exec=a_exec_np.astype(np.float32),
#             alpha=float(alpha),
#             label_safe=int(label_safe),
#             h_t=float(h_exe),
#             h_tp1=float(h_tp1),
#         )
#         self.primary.push(item)
#         (self.safe_buf if label_safe==1 else self.unsafe_buf).push(item)

#         self.global_step += 1

#         return {"t": t, "reward": float(reward), "done": bool(done), "label_safe": int(label_safe),
#                 "h_pi": float(h_pi), "h_qp": float(h_qp), "h_exec": float(h_exe), "alpha": float(alpha)}

#     # ---- sampling / training ----
#     def sample_minibatch(self, batch_size:int) -> Dict[str, torch.Tensor]:
#         n_unsafe = min(len(self.unsafe_buf), batch_size//2)
#         n_safe   = batch_size - n_unsafe
#         items: List[ReplayItem] = []
#         if n_safe>0 and len(self.safe_buf)>0:   items += self.safe_buf.sample(n_safe)
#         if n_unsafe>0 and len(self.unsafe_buf)>0: items += self.unsafe_buf.sample(n_unsafe)
#         if len(items)==0: items = self.primary.sample(batch_size)
#         if len(items)==0: return {}

#         dev = self.device

#         def stack(attr): return torch.as_tensor(np.stack([getattr(it, attr) for it in items], axis=0),
#                                                dtype=torch.float32, device=dev)

#         batch = {
#             "state_seq":       stack("state_seq"),
#             "action_seq":      stack("action_seq"),
#             "state_seq_next":  stack("state_seq_next"),
#             "action_seq_next": stack("action_seq_next"),
#             "u_nom":           stack("u_nom"),
#             "u_pi":            stack("u_pi"),
#             "u_qp":            stack("u_qp"),
#             "u_exec":          stack("u_exec"),
#             "y":               torch.as_tensor(np.array([it.label_safe for it in items], dtype=np.float32), device=dev),
#         }
#         return batch

#     def train_step(self) -> Dict[str, float]:
#         batch = self.sample_minibatch(self.cfg.batch_size)
#         if not batch:
#             return {"loss": 0.0}

#         st   = batch["state_seq"]
#         ac   = batch["action_seq"]
#         stn  = batch["state_seq_next"]
#         acn  = batch["action_seq_next"]
#         u_nom= batch["u_nom"]
#         u_pi = batch["u_pi"]
#         u_qp = batch["u_qp"]
#         u_ex = batch["u_exec"]
#         y    = batch["y"]
#         dev  = self.device

#         # Forward backbone
#         out, _ = self.model.pact({"state": st, "action": ac})
#         last_state = out[:,0::2,:][:,-1,:]

#         # Policy current prediction
#         delta = self.model.policy(last_state)
#         u_pred = torch.clamp(u_nom + delta,
#                              min=torch.as_tensor(self.act_low,  device=dev),
#                              max=torch.as_tensor(self.act_high, device=dev))

#         # Critic at u_pred
#         h_pred = h_score(self.model, st, ac, u_pred)

#         # Critic losses
#         L_bce = nn.functional.binary_cross_entropy_with_logits(h_pred, y)
#         L_margin = ((1-y) * torch.relu(self.cfg.margin + h_pred) + (y) * torch.relu(self.cfg.margin - h_pred)).mean()

#         # grad penalty for smoother rectifier
#         u_pred_req = u_pred.detach().requires_grad_(True)
#         h_tmp = h_score(self.model, st, ac, u_pred_req)
#         g = torch.autograd.grad(h_tmp.sum(), u_pred_req, retain_graph=False, create_graph=False)[0]
#         L_grad = (g*g).sum(dim=-1).mean()

#         # QP teacher on the batch (anchor nominal)
#         u_qp_batch = qp_teacher_anchor_nominal(self.model, st, ac, u_nom, margin=self.cfg.margin, eps=self.cfg.grad_eps)

#         # Alpha
#         alpha = torch.tensor(self.alpha(), device=dev)

#         # Policy losses
#         #  - imitate QP on unsafe (weighted by 1-alpha)
#         #  - stay near nominal on safe
#         #  - consistency with executed mix (stabilizes training distribution)
#         L_teacher = ((u_pred - u_qp_batch)**2).sum(dim=-1) * (1.0 - y)
#         L_safe    = ((u_pred - u_nom    )**2).sum(dim=-1) * y
#         u_mix_det = ((1.0 - alpha)*u_qp_batch + alpha*u_pred.detach())
#         L_exec    = ((u_pred - u_mix_det)**2).sum(dim=-1)
#         L_pi = (1.0 - alpha) * L_teacher.mean() + self.cfg.w_pi_safe * L_safe.mean() + self.cfg.w_exec * L_exec.mean()

#         # hdot finite-difference trend (safe states):  dot{h} + lambda*h >= 0
#         h_k   = h_score(self.model, st,  ac,  u_ex)  # at time t, executed action
#         h_kp1 = h_score(self.model, stn, acn, u_ex)  # evaluate at t+1 repeating u_ex (cheap proxy)
#         hdot  = (h_kp1 - h_k) / max(self.cfg.dt, 1e-6)
#         resid = self.cfg.hdot_lambda * h_k + hdot
#         L_hdot = torch.relu(-resid) * y    # apply only to safe labels
#         L_hdot = L_hdot.mean()

#         loss = self.cfg.w_bce*L_bce + self.cfg.w_margin*L_margin + self.cfg.w_grad*L_grad + L_pi + self.cfg.w_hdot*L_hdot

#         self.opt.zero_grad(set_to_none=True)
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
#         self.opt.step()

#         return {
#             "loss": float(loss.item()),
#             "L_bce": float(L_bce.item()),
#             "L_margin": float(L_margin.item()),
#             "L_grad": float(L_grad.item()),
#             "L_pi": float(L_pi.item()),
#             "L_hdot": float(L_hdot.item()),
#         }

#     def run_episode(self, max_steps: Optional[int]=None) -> Dict[str, Any]:
#         max_steps = max_steps or self.cfg.max_step
#         obs = self.env.reset(seed=self.cfg.seed)
#         self.ctx.init(obs["full_obs"].astype(np.float32), np.zeros((2,), dtype=np.float32))

#         ep_reward = 0.0
#         logs = []
#         for t in range(max_steps):
#             step_out = self.step_once(t)
#             ep_reward += float(step_out["reward"])

#             train_out = {"loss": 0.0}
#             if self.cfg.update_every > 0 and (t % self.cfg.update_every == 0):
#                 train_out = self.train_step()

#             logs.append({**step_out, **train_out})
#             if self.cfg.verbose and (t % 20 == 0):
#                 print(f"[t={t:03d}] r={step_out['reward']:.3f} "
#                       f"h_pi={step_out['h_pi']:.3f} h_qp={step_out['h_qp']:.3f} h_exec={step_out['h_exec']:.3f} "
#                       f"alpha={step_out['alpha']:.2f} "
#                       f"label={step_out['label_safe']} loss={train_out.get('loss',0.0):.4f}")
#             if step_out["done"]:
#                 break

#         return {"ep_reward": ep_reward, "steps": t+1, "logs": logs}


# # -----------------------------
# # Convenience API
# # -----------------------------
# import os
# from typing import Optional, Dict, Any
# def run_online_episode(cfg: Optional[OnlineConfig]=None) -> Dict[str, Any]:
#     cfg = cfg or OnlineConfig()
#     os.environ["PACT_DEVICE"] = cfg.device
#     runner = OnlineRunner(cfg)
#     return runner.run_episode()


# # =============================
# # Main Entry Point
# # =============================
# def main():
#     """
#     Main function to run the online training loop.
    
#     Process:
#     1. Parse command-line arguments (if any)
#     2. Create configuration with default or user-specified parameters
#     3. Initialize the OnlineRunner (which sets up env, model, buffers)
#     4. Run training episodes in a loop
#     5. Log progress after each episode
    
#     The agent learns continuously as it interacts with the environment,
#     with no separate data collection or offline training phase.
#     """
#     # Create configuration with default hyperparameters
#     cfg = OnlineConfig()
#     print("Config:", cfg)

#     # Initialize the training runner
#     # This creates the environment, model, optimizer, and buffers
#     runner = OnlineRunner(cfg)

#     # === Main Training Loop ===
#     # Run multiple episodes to train the agent
#     for episode in range(5000):
#         # Run one complete episode (agent acts until done or max_step reached)
#         # The runner handles:
#         # - Resetting the environment
#         # - Getting actions from policy/teacher
#         # - Executing actions in the environment
#         # - Storing experiences in replay buffer
#         # - Triggering training updates
#         ep_out = runner.run_episode()
        
#         # Log episode statistics
#         print(f"Ep {episode}: Steps={ep_out['steps']}, Reward={ep_out['ep_reward']:.2f}")


# if __name__ == "__main__":
#     main()
