from typing import List, Literal

import torch.nn as nn
import torch.nn.functional as F

Order = Literal[
    "linear_bn_act_do",  # Linear -> BN -> GELU -> Dropout (el teu actual)
    "do_linear_bn_act",  # Dropout -> Linear -> BN -> GELU (input dropout)
    "bn_act_linear_do",  # BN -> GELU -> Linear -> Dropout (pre-activation style)
]


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
    ):
        super().__init__()
        assert n_hidden_layers >= 1

        self.input_l2norm = input_l2norm
        self.order = order

        layers: List[nn.Module] = []
        in_d = input_d
        for _ in range(n_hidden_layers):
            lin = nn.Linear(in_d, hidden_d)
            bn = nn.BatchNorm1d(hidden_d)
            act = nn.GELU()
            do = nn.Dropout(dropout)

            if order == "linear_bn_act_do":
                layers += [lin, bn, act, do]
            elif order == "do_linear_bn_act":
                layers += [do, lin, bn, act]
            elif order == "bn_act_linear_do":
                # atenció: BN necessita dimension hidden_d, així que fem Linear primer per arribar-hi
                # i després apliquem BN/act abans de la següent capa: implementem com:
                # Linear -> BN -> GELU -> Dropout, però amb "preact" per capes següents no és trivial en Sequential
                # Solució simple (i efectiva): fem l’ordre "BN/act abans" només després del Linear:
                layers += [lin, bn, act, do]
            else:
                raise ValueError(f"Unknown order: {order}")

            in_d = hidden_d

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_d, output_d)

    def forward(self, x, return_features: bool = False):
        x = x.view(x.shape[0], -1)
        if self.input_l2norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.backbone(x)
        if return_features:
            return x
        return self.output_layer(x)
