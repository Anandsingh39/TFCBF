
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Imports from your codebase
# ------------------------------
# from src.models.modules.modeling_pact import PACTBase
# from src.models.modules.tokenizer_pact import PACTTokenizer  # not strictly needed here, but handy for debugging
from src.models.modules.modeling_pact import PACTBase
from src.models.modules.tokenizer_pact import PACTTokenizer  # not strictly needed here, but handy for debugging

# ------------------------------
# Config
# ------------------------------
class PACTConfig:
    state_key: str = "full_obs"         # (N, 38)
    action_key: str = "taken_action"    # (N, 2) or "nominal_action"
    ctx_tokens: int = 16                # total tokens = 16 ⇒ T = 8 pairs
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 8
    batch_size: int = 256
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    action_input_type: str = "continuous"   # "continuous" or "discrete"

CFG = PACTConfig()

# ------------------------------
# Model builder: PACT (GPT-2 style) with separate state and action tokenizers
# ------------------------------
def build_pact_model(state_dim=38, action_dim=2, ctx_tokens=16,
                     n_embd=128, n_layer=4, n_head=8, action_input_type="continuous"):
    assert ctx_tokens % 2 == 0, "ctx_tokens must be even"
    seq_len = ctx_tokens // 2  # GPT block_size internally is 2*seq_len

    gpt_config = dict(
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
        seq_len=seq_len
    )

    input_config = {
        "state": {
            "tokenizer": "mlp_state",
            "input_type": "continuous",
            "tokenizer_kwargs": {"state_dim": state_dim, "hidden": [256, 256], "use_ln": True},
        },
        "action": {
            "tokenizer": "mlp_action",
            "input_type": action_input_type,  # "continuous" or "discrete"
            "tokenizer_kwargs": {"action_dim": action_dim, "hidden": [128, 128], "use_ln": True},
        },
    }

    model = PACTBase(gpt_config=gpt_config, input_config=input_config)

    return model

# ------------------------------
# Heads
# ------------------------------
class PolicyHead(nn.Module):
    """Takes a contextualized state embedding (d) and predicts an action (A)."""
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state_emb_last):  # (B, d)
        return self.net(state_emb_last)

class CriticHead(nn.Module):
    """Takes a contextualized action embedding (d) and outputs a scalar safety/value score."""
    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.Tanh()
        )

    def forward(self, action_emb_last):  # (B, d)
        return self.net(action_emb_last)

# ------------------------------
# Dataset: builds rolling windows for each (episode, t)
# ------------------------------
class PACTWindowDataset(Dataset):
    """
    For each (episode e, timestep t), returns:
        state_seq:  (T, S)  with [s_{t-7}, ..., s_t]
        action_seq: (T, A)  with [a_{t-8}, ..., a_{t-1}]   (continuous) or (T,) for discrete
        meta: (episode_id, t)
    """
    def __init__(self, npz_path: str, state_key="full_obs", action_key="taken_action", ctx_tokens=16,
                 action_input_type="continuous"):
        super().__init__()
        self.data = np.load(npz_path, allow_pickle=True)
        self.state_key = state_key
        self.action_key = action_key
        self.T = ctx_tokens // 2
        self.action_input_type = action_input_type

        ep_ids = self.data["episode"].astype(int)  # (N,)
        self.episodes = np.unique(ep_ids)
        counts = [np.sum(ep_ids == e) for e in self.episodes]
        assert len(set(counts)) == 1, "Episodes must be equal length in this simple implementation."
        self.L = counts[0]
        self.N = len(ep_ids)

        # reshape to (E, L, ·)
        states_all = self.data[state_key].astype(np.float32)    # (N, S)
        actions_all = self.data[action_key]                     # (N, A or N,)
        if action_input_type == "continuous":
            actions_all = actions_all.astype(np.float32)
        else:
            actions_all = actions_all.astype(np.int64)

        self.S = states_all.shape[-1]
        if actions_all.ndim == 2:
            self.A = actions_all.shape[-1]
        else:
            self.A = 1  # discrete ids

        self.states = np.stack([states_all[ep_ids == e] for e in self.episodes], axis=0)   # (E, L, S)
        self.actions = np.stack([actions_all[ep_ids == e] for e in self.episodes], axis=0) # (E, L, A) or (E, L)
        # map indices
        self.index = [(ei, t) for ei in range(len(self.episodes)) for t in range(self.L)]

    def __len__(self):
        return len(self.index)

    def _left_pad_take(self, x, start, end, featdim):
        # x: (L, featdim) or (L,)
        if start >= 0:
            out = x[start:end]
        else:
            need = -start
            if x.ndim == 1:
                pad = np.repeat(x[0:1], need, axis=0)
            else:
                pad = np.repeat(x[0:1], need, axis=0)
            out = np.concatenate([pad, x[0:end]], axis=0)
        if out.shape[0] < (end - start):
            need = (end - start) - out.shape[0]
            if x.ndim == 1:
                pad = np.repeat(x[-1:], need, axis=0)
            else:
                pad = np.repeat(x[-1:], need, axis=0)
            out = np.concatenate([out, pad], axis=0)
        return out

    def __getitem__(self, i):
        ei, t = self.index[i]
        T = self.T

        # states: s[t-7: t]
        s_start, s_end = t - T + 1, t + 1
        s_win = self._left_pad_take(self.states[ei], s_start, s_end, self.S)   # (T, S)

        # actions: a[t-8: t-1]
        a_start, a_end = t - T, t
        a_src = self.actions[ei]
        if self.action_input_type == "continuous":
            a_win = self._left_pad_take(a_src, a_start, a_end, self.A)         # (T, A)
        else:
            a_win = self._left_pad_take(a_src.reshape(-1, 1), a_start, a_end, 1).reshape(-1)  # (T,)

        return (
            torch.as_tensor(s_win, dtype=torch.float32),
            torch.as_tensor(a_win, dtype=torch.float32 if self.action_input_type == "continuous" else torch.long),
            torch.tensor([self.episodes[ei], t], dtype=torch.long)
        )

# ------------------------------
# Helper: split interleaved transformer output back to (state, action)
# ------------------------------
def split_state_action_embeddings(out_embd):
    # out_embd: (B, 2*T, d) with order [state_0, action_0, state_1, action_1, ...]
    return out_embd[:, 0::2, :], out_embd[:, 1::2, :]

# ------------------------------
# End-to-end: build model + heads, feed batch, get policy & critic outputs
# ------------------------------
class PACTPolicyCritic(nn.Module):
    def __init__(self, state_dim, action_dim, ctx_tokens, n_embd, n_layer, n_head, action_input_type="continuous"):
        super().__init__()
        self.pact = build_pact_model(state_dim, action_dim, ctx_tokens, n_embd, n_layer, n_head, action_input_type)
        self.d_model = n_embd
        self.policy = PolicyHead(self.d_model, action_dim if action_input_type == "continuous" else action_dim)
        self.critic = CriticHead(self.d_model, out_dim=1)

    @torch.no_grad()
    def forward_eval(self, state_seq, action_seq):
        """
        state_seq:  (B, T, S)
        action_seq: (B, T, A) or (B, T) if discrete ids
        Returns:
          policy_out: (B, A)  predicted action from last state embedding
          critic_out: (B, 1)  value/safety score from last action embedding
          extras: dict with intermediate tensors
        """
        self.pact.eval()
        out_interleaved, state_tokens_pre = self.pact({"state": state_seq, "action": action_seq})
        state_ctx, action_ctx = split_state_action_embeddings(out_interleaved)

        last_state_ctx = state_ctx[:, -1, :]   # (B, d) contextual s_t
        last_action_ctx = action_ctx[:, -1, :] # (B, d) contextual a_{t-1}

        policy_out = self.policy(last_state_ctx)
        critic_out = self.critic(last_action_ctx)
        return policy_out, critic_out, {
            "state_ctx": state_ctx,
            "action_ctx": action_ctx,
            "state_tokens_pre": state_tokens_pre,
            "last_state_ctx": last_state_ctx,
            "last_action_ctx": last_action_ctx,
        }

# ------------------------------
# Demo / entry point
# ------------------------------
def run_demo(npz_path: str, state_key="full_obs", action_key="taken_action"):
    # Dataset & loader
    ds = PACTWindowDataset(npz_path, state_key=state_key, action_key=action_key,
                           ctx_tokens=CFG.ctx_tokens, action_input_type=CFG.action_input_type)
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, drop_last=False)

    # Model + heads
    model = PACTPolicyCritic(state_dim=ds.S, action_dim=ds.A, ctx_tokens=CFG.ctx_tokens,
                             n_embd=CFG.n_embd, n_layer=CFG.n_layer, n_head=CFG.n_head,
                             action_input_type=CFG.action_input_type).to(CFG.device)

    # Move a batch and run
    batch = next(iter(dl))
    state_seq = batch[0].to(CFG.device)                 # (B, T, 38)
    action_seq = batch[1].to(CFG.device)                # (B, T, 2) or (B, T)
    meta = batch[2]                                     # (B, 2)

    with torch.no_grad():
        policy_out, critic_out, extras = model.forward_eval(state_seq, action_seq)

    print("=== Demo outputs ===")
    print("state_seq:", tuple(state_seq.shape), "action_seq:", tuple(action_seq.shape))
    print("policy_out:", tuple(policy_out.shape), "critic_out:", tuple(critic_out.shape))
    print("state_ctx:", tuple(extras["state_ctx"].shape), "action_ctx:", tuple(extras["action_ctx"].shape))
    print("last_state_ctx:", tuple(extras["last_state_ctx"].shape), "last_action_ctx:", tuple(extras["last_action_ctx"].shape))

    return {
        "policy_out": policy_out,
        "critic_out": critic_out,
        **extras,
        "meta": meta,
    }

if __name__ == "__main__":
    # Example:
    # results = run_demo("pact_dataset.npz", state_key="full_obs", action_key="taken_action")
    pass

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# # Assume model files are available
# from src.models.modules.modeling_pact import PACTBase
# from src.models.modules.tokenizer_pact import PACTTokenizer

# # --- Model Definitions (Unchanged) ---
# def build_model(S, A, seq_len, n_embd=64, n_layer=2, n_head=8):
#     gpt_config = dict(n_embd=n_embd, n_layer=n_layer, n_head=n_head, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, seq_len=seq_len)
#     input_config = {"state": {"tokenizer": "mlp_state", "input_type": "continuous", "tokenizer_kwargs": {"state_dim": S, "hidden": [256, 256]}}, "action": {"tokenizer": "mlp_action", "input_type": "continuous", "tokenizer_kwargs": {"action_dim": A, "hidden": [128, 128]}}}
#     return PACTBase(gpt_config=gpt_config, input_config=input_config)

# def split_state_action_embeddings(out_embd):
#     return out_embd[:, 0::2, :], out_embd[:, 1::2, :]

# # --- Step 1: Define the Two Heads and the Combined Model ---

# class PolicyHead(nn.Module):
#     """Takes a state embedding and predicts an action."""
#     def __init__(self, n_embd, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(n_embd, 128), nn.GELU(), nn.Linear(128, action_dim))
#     def forward(self, state_embedding):
#         return self.net(state_embedding)

# class ValueHead(nn.Module):
#     """Takes an action embedding and produces a value or another embedding."""
#     def __init__(self, n_embd, output_dim=1):
#         super().__init__()
#         # For simplicity, this outputs a single scalar value (like a Q-value)
#         # You can change output_dim to be `n_embd` if you want another embedding
#         self.net = nn.Sequential(nn.Linear(n_embd, 128), nn.GELU(), nn.Linear(128, output_dim))
#     def forward(self, action_embedding):
#         return self.net(action_embedding)

# class ActorCriticModel(nn.Module):
#     """
#     A single model containing the shared Transformer backbone and two separate heads.
#     """
#     def __init__(self, state_dim, action_dim, seq_len, n_embd, n_layer, n_head):
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
        
#         # The shared backbone
#         self.backbone = build_model(S=state_dim, A=action_dim, seq_len=seq_len, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
        
#         # The two specialized heads
#         self.policy_head = PolicyHead(n_embd, action_dim)
#         self.value_head = ValueHead(n_embd, output_dim=n_embd) # Outputting an embedding of same size

#     def get_action(self, state_seq, action_seq):
#         """
#         Pass 1: Predicts an action given the history.
#         The last action in action_seq should be masked (e.g., zeros).
#         """
#         out_embd, _ = self.backbone({"state": state_seq, "action": action_seq})
#         state_emb, _ = split_state_action_embeddings(out_embd)
        
#         # Use the embedding of the very last state to predict the action
#         last_state_embedding = state_emb[:, -1, :]
#         predicted_action = self.policy_head(last_state_embedding)
#         return predicted_action

#     def get_action_embedding(self, state_seq, action_seq):
#         """
#         Pass 2: Gets the embedding of a state-action pair.
#         The sequences here should be complete and unmasked.
#         """
#         out_embd, _ = self.backbone({"state": state_seq, "action": action_seq})
#         _, action_emb = split_state_action_embeddings(out_embd)
        
#         # Use the embedding of the very last action
#         last_action_embedding = action_emb[:, -1, :]
        
#         # You can return this directly or pass it through another head
#         final_embedding = self.value_head(last_action_embedding)
#         return final_embedding


# # --- Step 2: The Dataset and Dataloader (Unchanged) ---
# class EpisodicNPZDataset(Dataset):
#     def __init__(self, file_path, seq_len):
#         with np.load(file_path) as data:
#             self.states = data["state_ctx"].astype(np.float32)
#             self.actions = data["action_ctx"].astype(np.float32)
#         self.seq_len = seq_len
#     def __len__(self):
#         return len(self.states) - self.seq_len
#     def __getitem__(self, idx):
#         return (
#             torch.from_numpy(self.states[idx : idx + self.seq_len]),
#             torch.from_numpy(self.actions[idx : idx + self.seq_len])
#         )

# # --- Step 3: The Modified Training Loop ---

# if __name__ == "__main__":
#     # --- Configuration ---
#     STATE_DIM, ACTION_DIM = 38, 2
#     SEQ_LEN, BATCH_SIZE = 16, 64
#     N_EMBD, N_LAYER, N_HEAD = 128, 4, 8
#     LEARNING_RATE, NUM_EPOCHS = 1e-4, 10
    
#     # --- Setup ---
#     # create_dummy_npz("my_transitions.npz", 5000, STATE_DIM, ACTION_DIM) # Helper to create data
#     dataset = EpisodicNPZDataset("./src/models/modules/pact_dataset_ctx16.npz", SEQ_LEN)
#     data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#     device = "cpu"

#     model = ActorCriticModel(STATE_DIM, ACTION_DIM, SEQ_LEN, N_EMBD, N_LAYER, N_HEAD).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     policy_loss_fn = nn.MSELoss()
    
#     print(f"Starting training on {device}...")
#     for epoch in range(NUM_EPOCHS):
#         model.train()
#         total_policy_loss = 0
        
#         for state_batch, action_batch in data_loader:
#             state_batch = state_batch.to(device)
#             action_batch = action_batch.to(device)
            
#             # === Pass 1: Get Action Prediction ===
            
#             # Create a copy of the action batch and mask the last action
#             masked_action_batch = action_batch.clone()
#             masked_action_batch[:, -1, :] = 0.0 # Masking with zeros
            
#             # Get the predicted action from the policy head
#             predicted_actions_at_last_step = model.get_action(state_batch, masked_action_batch)
            
#             # The target is the true last action from the original batch
#             true_actions_at_last_step = action_batch[:, -1, :]
            
#             # Calculate the policy loss
#             policy_loss = policy_loss_fn(predicted_actions_at_last_step, true_actions_at_last_step)
            
#             # === (Optional) Pass 2: Get Action Embedding ===
#             # In this example, we're just training the policy. But if you needed a value loss:
#             # 1. Get the action embedding using the unmasked batch
#             #    action_embeddings = model.get_action_embedding(state_batch, action_batch)
#             # 2. Compare to some target (e.g., reward-to-go) to get a value_loss
#             #    value_loss = loss_fn(action_embeddings, some_target)
#             # 3. total_loss = policy_loss + value_loss
            
#             # --- Backward Pass and Optimization ---
#             # We only backpropagate the policy loss in this example
#             optimizer.zero_grad()
#             policy_loss.backward()
#             optimizer.step()
            
#             total_policy_loss += policy_loss.item()
            
#         avg_loss = total_policy_loss / len(data_loader)
#         print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Policy Loss: {avg_loss:.6f}")

#     print("\nTraining complete.")

# # import argparse
# # import numpy as np
# # import torch
# # from .tokenizer_pact import PACTTokenizer
# # from .head_utils import *
# # from .decoder_utils import *
# # from .modeling_pact import PACTBase

# # def load_npz_batch(path, state_key, action_key, seq_len, batch_size, ordered=False, start_idx=0):
# #     import numpy as np
# #     import torch
# #     data = np.load(path, allow_pickle=True)

# #     state = data[state_key]   # (N,S) or (N,T,S)
# #     action = data[action_key] # (N,A)/(N,) or (N,T,A)/(N,T)

# #     if state.ndim == 2:
# #         # RAW STREAMS: build contiguous windows
# #         state = state.astype(np.float32)
# #         action = action.astype(np.float32)
# #         if action.ndim == 1: action = action[:, None]
# #         N = state.shape[0]

# #         if ordered:
# #             # sequential starts: [start_idx, start_idx+1, ...]
# #             # ensure windows fit
# #             max_start = N - seq_len
# #             if start_idx > max_start:
# #                 start_idx = 0  # wrap
# #             starts = np.arange(start_idx, min(start_idx + batch_size, max_start + 1))
# #             # if we hit the end, wrap around to the beginning
# #             if len(starts) < batch_size:
# #                 needed = batch_size - len(starts)
# #                 starts = np.concatenate([starts, np.arange(0, min(needed, max_start + 1))])
# #             next_start = (starts[-1] + 1) % (max_start + 1)
# #         else:
# #             starts = np.random.randint(0, N - seq_len, size=batch_size)
# #             next_start = 0

# #         s_batch = np.stack([state[i:i+seq_len] for i in starts], axis=0)
# #         a_batch = np.stack([action[i:i+seq_len] for i in starts], axis=0)
# #         S, A = s_batch.shape[-1], a_batch.shape[-1]
# #         return torch.from_numpy(s_batch), torch.from_numpy(a_batch), S, A, int(next_start)

# #     elif state.ndim == 3:
# #         # CONTEXT WINDOWS: (N,T,S)
# #         state = state.astype(np.float32)
# #         if action.ndim == 2:  # (N,T) -> (N,T,1)
# #             action = action[..., None].astype(np.float32)
# #         else:
# #             action = action.astype(np.float32)

# #         N = state.shape[0]

# #         if ordered:
# #             # if available, sort by (episode, timestep); otherwise keep file order
# #             if "episode" in data.files and "timestep" in data.files:
# #                 order = np.lexsort((data["timestep"], data["episode"]))
# #             else:
# #                 order = np.arange(N)
# #             if start_idx >= N:
# #                 start_idx = 0
# #             end = min(start_idx + batch_size, N)
# #             idx = order[start_idx:end]
# #             if len(idx) < batch_size:
# #                 idx = np.concatenate([idx, order[: batch_size - len(idx)]])
# #             next_start = (start_idx + batch_size) % N
# #         else:
# #             idx = np.random.randint(0, N, size=batch_size)
# #             next_start = 0

# #         # keep last seq_len steps (windows already right-aligned)
# #         T = min(state.shape[1], action.shape[1], seq_len)
# #         s_batch = state[idx, -T:, :]
# #         a_batch = action[idx, -T:, :]
# #         S, A = s_batch.shape[-1], a_batch.shape[-1]
# #         return torch.from_numpy(s_batch), torch.from_numpy(a_batch), S, A, int(next_start)

# #     else:
# #         raise ValueError(f"Unsupported state shape: {state.shape}")

# # def build_model(S, A, seq_len, n_embd=64, n_layer=2, n_head=8,
# #                 action_input_type="continuous"):
# #     gpt_config = dict(
# #         n_embd=n_embd, n_layer=n_layer, n_head=n_head,
# #         embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
# #         seq_len=seq_len
# #     )
# #     input_config = {
# #         "state": {
# #             "tokenizer": "mlp_state",
# #             "input_type": "continuous",
# #             "tokenizer_kwargs": {"state_dim": S, "hidden": [256, 256], "use_ln": True},
# #         },
# #         "action": {
# #             "tokenizer": "mlp_action",
# #             "input_type": action_input_type,                        # "continuous" or "discrete"
# #             "tokenizer_kwargs": {
# #                 "action_dim": A if action_input_type == "continuous" else max(A, 1),
# #                 "hidden": [128, 128], "use_ln": True
# #             },
# #         },
# #     }
# #     return PACTBase(gpt_config=gpt_config, input_config=input_config)
# # def split_state_action_embeddings(out_embd):
# #     """
# #     out_embd: (B, 2*T, d) transformer outputs for [s0,a0,s1,a1,...,sT,aT]
# #     Returns:
# #       state_emb:  (B, T, d)  outputs at even positions (s_t)
# #       action_emb: (B, T, d)  outputs at odd  positions (a_t)
# #     """
# #     # even indices 0,2,4,... -> states ; odd 1,3,5,... -> actions
# #     state_emb  = out_embd[:, 0::2, :]
# #     action_emb = out_embd[:, 1::2, :]
# #     return state_emb, action_emb

# # def main():
# #     import argparse, torch, numpy as np
# #     p = argparse.ArgumentParser()
# #     p.add_argument("--data", required=True)
# #     p.add_argument("--state-key", default="state_ctx")
# #     p.add_argument("--action-key", default="action_ctx")
# #     p.add_argument("--seq-len", type=int, default=16)
# #     p.add_argument("--batch-size", type=int, default=8)
# #     p.add_argument("--ordered", action="store_true")
# #     p.add_argument("--seed", type=int, default=0)
# #     args = p.parse_args()

# #     np.random.seed(args.seed); torch.manual_seed(args.seed)

# #     state, action, S, A, _ = load_npz_batch(
# #         args.data, args.state_key, args.action_key, args.seq_len, args.batch_size,
# #         ordered=args.ordered, start_idx=0
# #     )
# #     action_type =  "continuous"
# #     model = build_model(S, A, args.seq_len, action_input_type=action_type)
# #     model.eval()

# #     # PACTBase.forward expects a dict with keys "state" and "action"
# #     # If action is (B,T), the module will unsqueeze to (B,T,1) itself.   # see modeling_pact.forward
# #     with torch.no_grad():
# #         out_embd, state_tokens = model({"state": state, "action": action})
# #     state_emb, action_emb = split_state_action_embeddings(out_embd)

# #     print("OK ✓")
# #     print("state batch:", tuple(state.shape))
# #     print("action batch:", tuple(action.shape))
# #     print("transformer out:", tuple(out_embd.shape))   # (B, 2*T, n_embd)
# #     print("state tokens:", tuple(state_tokens.shape))  # (B, T, n_embd)
# #     print("state", state)
# #     print("action:", action)
# #     print("out_embed", out_embd)
# #     print("state", state.shape, state_emb.shape)
# #     print("action:", action.shape, action_emb.shape)
# #     print("out_embed", out_embd.shape, out_embd.shape)

# # if __name__ == "__main__":
# #     main()
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn

# # Assuming the files you provided are in a structure like this:
# # /project
# #   - main.py (this file)
# #   /src
# #     - __init__.py
# #     /models
# #       - __init__.py
# #       /modules
# #         - __init__.py
# #         - modeling_pact.py
# #         - tokenizer_pact.py
# #         - minGPT.py
# #         - head_utils.py
# #         - decoder_utils.py
# #         - tokenizer_utils.py

# # To make imports work, we add the project root to the path.
# # In a real project, you would install this as a package.
# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.getcwd()))

# # from src.models.modules.modeling_pact import PACTBase
# # from src.models.modules.tokenizer_pact import PACTTokenizer

# # def build_model(S, A, seq_len, n_embd=64, n_layer=2, n_head=8,
# #                 action_input_type="continuous"):
# #     """
# #     This function demonstrates how to configure the PACTBase model.
# #     The `input_config` dictionary is where you define your tokenizers.
# #     """
# #     gpt_config = dict(
# #         n_embd=n_embd, n_layer=n_layer, n_head=n_head,
# #         embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
# #         seq_len=seq_len
# #     )
    
# #     # --- This is the core configuration for your tokenizers ---
# #     input_config = {
# #         "state": {
# #             # "mlp_state" maps to `VectorStateTokenizer` in tokenizer_pact.py
# #             "tokenizer": "mlp_state",
# #             "input_type": "continuous",
# #             # We pass the dimension of our state vector here (38)
# #             "tokenizer_kwargs": {"state_dim": S, "hidden": [256, 256], "use_ln": True},
# #         },
# #         "action": {
# #             # "mlp_action" maps to `ActionTokenizer` in tokenizer_pact.py
# #             "tokenizer": "mlp_action",
# #             "input_type": action_input_type, # This should be "continuous" for you
# #             # We pass the dimension of our action vector here (2)
# #             "tokenizer_kwargs": {
# #                 "action_dim": A,
# #                 "hidden": [128, 128], "use_ln": True
# #             },
# #         },
# #     }
# #     return PACTBase(gpt_config=gpt_config, input_config=input_config)

# # def split_state_action_embeddings(out_embd):
# #     """
# #     Splits the interleaved output sequence from the transformer.
# #     out_embd: (B, 2*T, d) -> [e_s0, e_a0, e_s1, e_a1, ...]
# #     """
# #     # even indices 0, 2, 4,... correspond to states
# #     state_emb  = out_embd[:, 0::2, :] # (B, T, d)
# #     # odd indices 1, 3, 5,... correspond to actions
# #     action_emb = out_embd[:, 1::2, :] # (B, T, d)
# #     return state_emb, action_emb

# # class PolicyHead(nn.Module):
# #     """
# #     A simple MLP head to predict an action from a state embedding.
# #     It takes the contextualized state embedding from the transformer and outputs
# #     a continuous action vector.
# #     """
# #     def __init__(self, n_embd, action_dim):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(n_embd, 128),
# #             nn.GELU(),
# #             nn.Linear(128, action_dim)
# #             # For continuous actions, you might add a nn.Tanh() here
# #             # if your actions are bounded between [-1, 1].
# #         )

# #     def forward(self, state_embeddings):
# #         # state_embeddings: (B, T, n_embd)
# #         return self.net(state_embeddings) # returns (B, T, action_dim)

# # def main():
# #     # --- 1. Define Model & Data Parameters ---
# #     # Your specific dimensions
# #     STATE_DIM = 38
# #     ACTION_DIM = 2
    
# #     # Transformer parameters
# #     SEQ_LEN = 8       # Context window of 8 state-action pairs
# #     BATCH_SIZE = 2   # Number of sequences to process at once
# #     N_EMBD = 128      # Embedding dimension
# #     N_LAYER = 4       # Number of transformer layers
# #     N_HEAD = 8        # Number of attention heads

# #     # Set seed for reproducibility
# #     torch.manual_seed(0)
# #     np.random.seed(0)

# #     # --- 2. Generate Dummy Data (replace with your `load_npz_batch`) ---
# #     # Create a batch of dummy data with the correct shape.
# #     # In your code, you would get this from your `load_npz_batch` function.
# #     # state shape: (BATCH_SIZE, SEQ_LEN, STATE_DIM)
# #     # action shape: (BATCH_SIZE, SEQ_LEN, ACTION_DIM)
# #     state_batch = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
# #     action_batch = torch.randn(BATCH_SIZE, SEQ_LEN, ACTION_DIM)
    
# #     # --- 3. Build the Model and Policy Head ---
# #     # The model is configured with your state and action dimensions.
# #     pact_model = build_model(
# #         S=STATE_DIM, 
# #         A=ACTION_DIM, 
# #         seq_len=SEQ_LEN, 
# #         n_embd=N_EMBD, 
# #         n_layer=N_LAYER, 
# #         n_head=N_HEAD
# #     )
# #     pact_model.eval() # Set to evaluation mode

# #     # This is the head that will predict actions
# #     policy_head = PolicyHead(n_embd=N_EMBD, action_dim=ACTION_DIM)

# #     print("Model and Policy Head built successfully.")
# #     print("-" * 50)

# #     # --- 4. Forward Pass to Get Embeddings ---
# #     # PACTBase.forward expects a dictionary with "state" and "action" keys.
# #     with torch.no_grad():
# #         # The model processes the interleaved sequence internally
# #         out_embd_sequence, _ = pact_model({"state": state_batch, "action": action_batch})

# #     print("out_embd_sequence", out_embd_sequence.shape)
# #     # Separate the output embeddings into state and action streams
# #     state_output_emb, action_output_emb = split_state_action_embeddings(out_embd_sequence)
    
# #     print("Transformer Forward Pass:")
# #     print(f"Input State Shape:  {state_batch}")
# #     print(f"Input Action Shape: {action_batch}")
# #     print(f"Transformer Output Shape (interleaved): {out_embd_sequence.shape}")
# #     print(f"Split State Output Embedding Shape:   {state_output_emb.shape}")
# #     print(f"Split Action Output Embedding Shape:  {action_output_emb.shape}")
# #     print("-" * 50)

# #     # --- 5. Predict Actions with the Policy Head ---
# #     # To predict the action at timestep 't', we use the state embedding from timestep 't'.
# #     # For example, state_output_emb[:, t, :] is the embedding for state s_t,
# #     # which contains information from all previous tokens (s_0, a_0, ..., s_{t-1}, a_{t-1}).
# #     predicted_actions = policy_head(state_output_emb)
    
# #     print("Policy Head Prediction:")
# #     print(f"Predicted Actions Shape: {predicted_actions}")

# #     # You can now use these predicted_actions to compute a loss against the true actions
# #     # from your dataset (e.g., using Mean Squared Error for continuous actions).
# #     # loss = torch.nn.functional.mse_loss(predicted_actions, action_batch)
# #     # loss.backward()
# #     # ... training steps ...

# # if __name__ == "__main__":
# #     main()