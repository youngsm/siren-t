from typing import List
import torch
from torch import nn
import numpy as np
from slar.base import SineLayer

class SirenT(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        hidden_layers: List[int],
        out_features: List[int],
        outermost_linear: bool = False,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()

        hidden_features = [hidden_features] if isinstance(hidden_features, int) else hidden_features
        hidden_layers = [hidden_layers] if isinstance(hidden_layers, int) else hidden_layers
        out_features = [out_features] if isinstance(out_features, int) else out_features

        print(
            f"[Siren] {in_features} in => {out_features} out, hidden {hidden_features} features {hidden_layers} layers"
        )
        print(
            f"        omega {first_omega_0} first {hidden_omega_0} hidden, the final layer linear {outermost_linear}"
        )

        module0_hf, *moduleN_hf = hidden_features
        module0_hl, *moduleN_hl = hidden_layers

        # Build module0
        self.module0 = nn.Sequential(
            SineLayer(
                in_features,
                module0_hf,
                is_first=True,
                omega_0=first_omega_0,
            ),
            *[
                SineLayer(
                    module0_hf,
                    module0_hf,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
                for _ in range(module0_hl)
            ],
        )

        if not moduleN_hf:
            if outermost_linear:
                final_linear = nn.Linear(module0_hf, out_features[0])
                with torch.no_grad():
                    final_linear.weight.uniform_(
                       -np.sqrt(6 / module0_hf) / hidden_omega_0,
                        np.sqrt(6 / module0_hf) / hidden_omega_0,
                    )
                self.module0.add_module("final_linear", final_linear)
            else:
                self.module0.add_module(
                    "final_sine",
                    SineLayer(
                        module0_hf,
                        out_features[0],
                        is_first=False,
                        omega_0=hidden_omega_0,
                    ),
                )

        else:
            # Build moduleN
            self.moduleN = nn.ModuleList(
                [
                    nn.Sequential(
                        SineLayer(
                            module0_hf,
                            hf,
                            is_first=False,
                            omega_0=hidden_omega_0,
                        ),
                        *[
                            SineLayer(hf, hf, is_first=False, omega_0=hidden_omega_0)
                            for _ in range(nf - 1)
                        ],
                    )
                    for nf, hf in zip(moduleN_hl, moduleN_hf)
                ]
            )
            for i, (hf, of) in enumerate(zip(moduleN_hf, out_features)):
                if outermost_linear:
                    final_linear = nn.Linear(hf, of)
                    with torch.no_grad():
                        final_linear.weight.uniform_(
                            -np.sqrt(6 / hf) / hidden_omega_0,
                            np.sqrt(6 / hf) / hidden_omega_0,
                        )
                    self.moduleN[i].add_module("final_linear", final_linear)
                else:
                    self.moduleN[i].add_module(
                        "final_sine",
                        SineLayer(
                            hf,
                            of,
                            is_first=False,
                            omega_0=hidden_omega_0,
                        ),
                    )

    def forward(self, coords, clone=False, concat=False):
        if clone:
            coords = coords.clone().detach().requires_grad_(True)
        stage0 = self.module0(coords)
        if not self.moduleN:
            return stage0
        return torch.cat([module(stage0) for module in self.moduleN], dim=1)

    def freeze_all_except_moduleNidx(self, idx):
        """Freeze all parameters except those in moduleN[idx]"""

        self.unfreeze_all()
        for param in self.module0.parameters():
            param.requires_grad = False
        for i, module in enumerate(self.moduleN):
            for param in module.parameters():
                param.requires_grad = i == idx

    def unfreeze_all(self):
        """Unfreeze all parameters in the network"""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Return only the parameters that require gradients"""
        return filter(lambda p: p.requires_grad, self.parameters())

    def print_trainable_params(self):
        """Print the names of trainable parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
                
    def __repr__(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory_mb = n_params * 4 / (1024 * 1024 ) # assume fp32
        return (
            f"{n_params:,} trainable parameters\n{memory_mb:2f} MB\n{super().__repr__()}"
        )