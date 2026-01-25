from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Order = Literal[
    "linear_bn_act_do",  # Linear -> BN -> ACT -> Dropout
    "do_linear_bn_act",  # Dropout -> Linear -> BN -> ACT
    "bn_act_linear_do",  # (aprox) Linear -> BN -> ACT -> Dropout
]
ActName = Literal["gelu", "relu"]


class SimpleModel(nn.Module):
    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        output_d: int,
        n_hidden_layers: int = 3,
        dropout: float = 0.6,
        order: Order = "linear_bn_act_do",
        input_l2norm: bool = False,
        activation: ActName = "gelu",
    ):
        super().__init__()
        assert n_hidden_layers >= 1

        self.input_l2norm = input_l2norm
        self.order = order

        Act = nn.GELU if activation == "gelu" else nn.ReLU

        layers: List[nn.Module] = []
        in_d = input_d
        for _ in range(n_hidden_layers):
            lin = nn.Linear(in_d, hidden_d)
            bn = nn.BatchNorm1d(hidden_d)
            act = Act()
            do = nn.Dropout(dropout)

            if order == "linear_bn_act_do":
                layers += [lin, bn, act, do]
            elif order == "do_linear_bn_act":
                layers += [do, lin, bn, act]
            elif order == "bn_act_linear_do":
                # aproximació simple estable
                layers += [lin, bn, act, do]
            else:
                raise ValueError(f"Unknown order: {order}")

            in_d = hidden_d

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_d, output_d)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          - logits if return_features=False
          - (logits, features) if return_features=True
        features are the backbone output: shape (B, hidden_d)
        """
        x = x.view(x.shape[0], -1)
        if self.input_l2norm:
            x = F.normalize(x, p=2, dim=1)

        feats = self.backbone(x)  # (B, hidden_d)
        logits = self.output_layer(feats)  # (B, output_d)

        if return_features:
            return logits, feats
        return logits
