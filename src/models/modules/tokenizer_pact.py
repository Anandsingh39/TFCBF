# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from typing import Dict, Union

import torch
import torch.nn as nn
from src.models.modules.tokenizer_utils import PCL_encoder as PointNet
from src.models.modules.tokenizer_utils import resnet18_custom


class PACTTokenizer(nn.Module):
    def __init__(self, input_config, n_embd) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.tokenizers = nn.ModuleDict(
            {
                input_name: self.assign_tokenizer(input)
                for input_name, input in input_config.items()
            }
        )
        self.input_config = input_config

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:

        res: dict[str, torch.Tensor] = {}
        for input_name, input_tensor in x.items():
            size = input_tensor.size()
            input_tensor = input_tensor.reshape(
                self.tokenizers[input_name].batch_input_size
            ).contiguous()
            res[input_name] = self.tokenizers[input_name](input_tensor).reshape(
                (size[0], size[1], self.n_embd)
            )
        return res

    def assign_tokenizer(self, input: Dict[str, Union[str, dict]]) -> nn.Module:
        """Construct a tokenizer given the input configuration.

        Args:
           input (Dict[str, Union[str, dict]]): a dictionary that specifies tokenier to be used
           for each input channel

        Raises:
            NotImplementedError: _description_

        Returns:
            nn.Module: tokenizer to be used for input channel
        """
        tokenizer = None
        # existing mapping ...
        if input["tokenizer"] == "mlp_state":
            return VectorStateTokenizer(
                n_embd=self.n_embd,
                state_dim=input["tokenizer_kwargs"]["state_dim"],
                hidden=tuple(input["tokenizer_kwargs"].get("hidden", (256, 256))),
                use_ln=bool(input["tokenizer_kwargs"].get("use_ln", True)),
            )
        elif input["tokenizer"] == "mlp_action":
            return ActionTokenizer(
                n_embd=self.n_embd,
                input_type=input["input_type"],                        # "continuous" | "discrete"
                action_dim=input["tokenizer_kwargs"]["action_dim"],
                hidden=tuple(input["tokenizer_kwargs"].get("hidden", (128, 128))),
                use_ln=bool(input["tokenizer_kwargs"].get("use_ln", True)),
            )
        elif input["tokenizer"] == "resnet18":
            tokenizer = ResNet18Tokenizer(
                self.n_embd, input["input_type"], **input["tokenizer_kwargs"]
            )
        elif input["tokenizer"] == "pointnet":
            tokenizer = PointNetTokenizer(
                self.n_embd, input["input_type"], **input["tokenizer_kwargs"]
            )
        elif input["tokenizer"] == "simple_action":
            tokenizer = SimpleActionTokenizer(
                self.n_embd, input["input_type"], **input["tokenizer_kwargs"]
            )
        elif input["tokenizer"] == "complex_action":
            tokenizer = ComplexActionTokenizer(
                self.n_embd, input["input_type"], **input["tokenizer_kwargs"]
            )
        else:
            raise NotImplementedError(
                f"Tokenizer {input['tokenizer']} type not implemented yet"
            )
        return tokenizer


class ResNet18Tokenizer(nn.Module):
    def __init__(self, n_embd: int, input_type: str, n_channel: int = 1):
        super().__init__()
        self.state_encoder = nn.Sequential(
            resnet18_custom(pretrained=False, clip_len=n_channel),
            nn.ReLU(),
            nn.Linear(1000, n_embd),
            nn.Tanh(),
        )
        self.n_channel = n_channel

    def forward(self, x):
        return self.state_encoder(x)

    @property
    def batch_input_size(self):
        # hard coded size for the tokenizer module
        return (-1, self.n_channel, 224, 224)


class PointNetTokenizer(nn.Module):
    def __init__(self, n_embd, input_type):
        super().__init__()
        self.state_encoder = PointNet(n_embd)

    def forward(self, x):
        return self.state_encoder(x)

    @property
    def batch_input_size(self):
        # hard coded size for the tokenizer module
        return (-1, 2, 720)


class SimpleActionTokenizer(nn.Module):
    def __init__(self, n_embd, input_type, action_dim=4):
        super().__init__()
        self.action_embeddings = (
            nn.Sequential(nn.Linear(1, n_embd))
            if input_type == "continuous"
            else nn.Sequential(nn.Embedding(action_dim, n_embd))
        )

    def forward(self, x):
        return self.action_embeddings(x)

    @property
    def batch_input_size(self):
        # hard coded size for the tokenizer module
        return (-1, 1)


class VectorStateTokenizer(nn.Module):
    """
    Tokenizes a vector state x_t ∈ R^{S} to an embedding e_t ∈ R^{n_embd}.
    Intended input to the module (after PACTTokenizer reshapes): (-1, S).
    """
    def __init__(self, n_embd: int, state_dim: int, hidden=(256, 256), use_ln: bool = True):
        super().__init__()
        self.state_dim = int(state_dim)
        self.norm = nn.LayerNorm(self.state_dim) if use_ln else nn.Identity()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, state_dim)
        return self.net(self.norm(x))

    @property
    def batch_input_size(self):
        # PACTTokenizer will reshape (B, T, S) -> (-1, S) before calling forward()
        return (-1, self.state_dim)


class ActionTokenizer(nn.Module):
    """
    Tokenizes actions to an embedding e^a_t ∈ R^{n_embd}.
    - continuous: a_t ∈ R^{A} via MLP
    - discrete:   a_t ∈ {0..A-1} via nn.Embedding
    """
    def __init__(
        self,
        n_embd: int,
        input_type: str,                # "continuous" or "discrete"
        action_dim: int,
        hidden=(128, 128),
        use_ln: bool = True,
    ):
        super().__init__()
        self.input_type = input_type
        self.action_dim = int(action_dim)

        if input_type == "discrete":
            self.embed = nn.Embedding(self.action_dim, n_embd)
        elif input_type == "continuous":
            self.norm = nn.LayerNorm(self.action_dim) if use_ln else nn.Identity()
            h1, h2 = hidden
            self.net = nn.Sequential(
                nn.Linear(self.action_dim, h1),
                nn.GELU(),
                nn.Linear(h1, h2),
                nn.GELU(),
                nn.Linear(h2, n_embd),
            )
        else:
            raise ValueError("input_type must be 'continuous' or 'discrete'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # continuous: x is (-1, A) float
        # discrete:   x is (-1,) long (class ids)
        if self.input_type == "discrete":
            if x.dim() != 1:
                x = x.view(-1)  # ensure shape (-1,)
            return self.embed(x.long())
        else:
            return self.net(self.norm(x))

    @property
    def batch_input_size(self):
        # PACTTokenizer uses this to reshape the incoming (B, T, ·)
        if self.input_type == "discrete":
            return (-1,)         # a single id per token
        else:
            return (-1, self.action_dim)


class ComplexActionTokenizer(nn.Module):
    def __init__(self, n_embd, input_type):
        super().__init__()
        self.action_embeddings = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_embd),
        )

    def forward(self, x):
        return self.action_embeddings(x)

    @property
    def batch_input_size(self):
        # hard coded size for the tokenizer module
        return (-1, 1)
