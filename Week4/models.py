from typing import Callable, List, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms.v2 as F
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor
from torchview import draw_graph
from torchvision import models


class SimpleCNNv1(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNv1, self).__init__()

        def block(in_c, out_c, dropout_rate=0.3):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_rate),
            )

        self.b1 = block(3, 32)  # 128 → 64
        self.b2 = block(32, 64)  # 64 → 32
        self.b3 = block(64, 128)  # 32 → 16

        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout_fc(functional.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class SimpleCNNv2(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()

        def block(in_c, out_c, dropout_rate=0.2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_rate),
            )

        self.b1 = block(3, 32)  # 64 → 32
        self.b2 = block(32, 16)  # 32 → 16
        # self.b3 = block(64, 128)  # 16 → 8

        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 16, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        # x = self.b3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout_fc(functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def extract_grad_cam(
        self,
        input_image: torch.Tensor,
        target_layer: List[Type[nn.Module]],
        targets: List[Type[ClassifierOutputTarget]],
    ) -> Type[GradCAMPlusPlus]:
        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam

    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False


class SimpleCNNv3(nn.Module):
    def __init__(self, image_input_size, num_classes, dropout_rate=0.2):
        super().__init__()

        def block(in_c, out_c, strides=1, dropout_rate=0.2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_rate),
            )

        self.b1_out = 32
        self.b2_out = 16
        self.fc1_out = 16
        self.fc1_input_size = (image_input_size // 4) * (image_input_size // 4)
        self.b1 = block(3, self.b1_out, dropout_rate=dropout_rate)  # 128 → 64
        self.b2 = block(self.b1_out, self.b2_out, dropout_rate=dropout_rate)  # 64 → 32

        self.skip1 = nn.Sequential(
            nn.Conv2d(
                3, self.b1_out, kernel_size=1, stride=2
            ),  # stride=2 to match MaxPool
        )
        self.skip1_relu = nn.ReLU()
        self.skip2 = nn.Sequential(
            nn.Conv2d(self.b1_out, self.b2_out, kernel_size=1, stride=2),
        )
        self.skip2_relu = nn.ReLU()

        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.fc1_input_size * self.b2_out, self.fc1_out)
        self.fc2 = nn.Linear(self.fc1_out, num_classes)

    def forward(self, x):
        previous_input = x
        out = self.b1(x)
        out = out + self.skip1(previous_input)
        out = self.skip1_relu(out)

        previous_input = out
        out = self.b2(out)
        out = out + self.skip2(previous_input)
        out = self.skip2_relu(out)

        out = out.view(out.size(0), -1)
        out = self.dropout_fc(functional.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

    def extract_grad_cam(
        self,
        input_image: torch.Tensor,
        target_layer: List[Type[nn.Module]],
        targets: List[Type[ClassifierOutputTarget]],
    ) -> Type[GradCAMPlusPlus]:
        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam

    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False


class SimpleCNNvAttn(nn.Module):
    def __init__(self, image_input_size, num_classes, dropout_rate=0.2, use_attn=True):
        super().__init__()

        def block(in_c, out_c, strides=1, dropout_rate=0.2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=strides, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(dropout_rate),
            )

        self.b1_out = 32
        self.b2_out = 32
        self.fc_input_size = image_input_size // 4
        self.b1 = block(3, self.b1_out, dropout_rate=dropout_rate)  # 128 → 64
        self.b2 = block(self.b1_out, self.b2_out, dropout_rate=dropout_rate)  # 64 → 32

        if use_attn:
            self.attn1 = SelfAtt(self.b1_out)
            self.attn2 = SelfAtt(self.b2_out)
        else:
            self.attn1 = None
            self.attn2 = None

        self.skip1 = nn.Sequential(
            nn.Conv2d(
                3, self.b1_out, kernel_size=1, stride=2
            ),  # stride=2 to match MaxPool
        )
        self.skip1_relu = nn.ReLU()
        self.skip2 = nn.Sequential(
            nn.Conv2d(self.b1_out, self.b2_out, kernel_size=1, stride=2),
        )
        self.skip2_relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.b2_out, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        previous_input = x
        out = self.b1(x)
        out = out + self.skip1(previous_input)
        out = self.skip1_relu(out)
        out = self.attn1(out) if self.attn1 is not None else out
        previous_input = out
        out = self.b2(out)
        out = out + self.skip2(previous_input)
        out = self.skip2_relu(out)
        out = self.attn2(out) if self.attn2 is not None else out

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)
        return out

    def extract_grad_cam(
        self,
        input_image: torch.Tensor,
        target_layer: List[Type[nn.Module]],
        targets: List[Type[ClassifierOutputTarget]],
    ) -> Type[GradCAMPlusPlus]:
        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam

    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False


# https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
class ShuffleCNNvAttn(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout_rate=0.2,
        use_attn=True,
        knowledge_distillation=False,
    ):
        super().__init__()

        self.b1_out = self.b2_out = self.b3_out = 16
        input_channels = 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.b1_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.b1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage2 = InvertedResidual(self.b1_out, self.b2_out, stride=2)
        self.stage3 = InvertedResidual(self.b2_out, self.b3_out, stride=2)

        if use_attn:
            self.attn1 = SelfAtt(self.b1_out)
            self.attn2 = SelfAtt(self.b2_out)
        else:
            self.attn1 = None
            self.attn2 = None

        if knowledge_distillation:
            self.regressor = nn.Sequential(
                nn.Conv2d(self.b3_out, 112, kernel_size=3, padding=1),
            )
        else:
            self.regressor = None

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.b3_out, self.b3_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.b3_out),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(self.b3_out, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage2(out)
        out = self.attn1(out) if self.attn1 is not None else out

        out = self.stage3(out)
        out = self.attn2(out) if self.attn2 is not None else out

        out = self.conv4(out)

        if self.regressor is not None:
            regressor_output = self.regressor(out)
        else:
            regressor_output = None

        out = out.mean([2, 3])

        out = self.fc(out)
        return out, regressor_output

    def extract_grad_cam(
        self,
        input_image: torch.Tensor,
        target_layer: List[Type[nn.Module]],
        targets: List[Type[ClassifierOutputTarget]],
    ) -> Type[GradCAMPlusPlus]:
        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam

    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, num_channels, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class SelfAtt(nn.Module):
    def __init__(self, in_ch):
        super(SelfAtt, self).__init__()
        self.weights = nn.Parameter(torch.zeros(1))
        self.key = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.query = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.value = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, intermediate=False):
        N, ch, h, w = x.shape
        # value = self.value(x).view(N, -1, h*w) # N, ch, h*w
        key = self.key(x).view(N, -1, h * w)  # N, ch, h*w
        query = self.query(x).view(N, -1, h * w).permute(0, 2, 1)  # N, h*w, ch

        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # torch.bmm(input, mat2, *, out=None) → Tensor
        # input - bxnxm and mat2 - bxmxp then out is bxnxp

        attention = self.softmax(
            # query is N, h*w, ch
            # key is N, ch, h*w
            # torch.bmm out -> N, h*w, h*w
            # att = smax(torch.bmm)
            torch.bmm(query, key)
        )

        value = self.value(x).view(N, -1, h * w)  # N, ch, h*w

        out = torch.bmm(
            value,  # N, ch, h*w
            attention.permute(0, 2, 1),  # N, h*w, h*w
        ).view(  # output of bmm is N, ch, h*w
            N, ch, h, w
        )

        # Residual connection
        # self.weights is a trainable param that further weighs contribution of attention
        out = self.weights * out + x

        # return based on inference
        if intermediate:
            return out, attention, query, key
        else:
            return out


class TeacherModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_extraction: bool = True,
        arch_type: str = "original",
        dropout_rate: float = 0.2,
        use_batchnorm: bool = False,
    ):
        super(TeacherModel, self).__init__()

        # Load pretrained EfficientNet-B0 model
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        if feature_extraction:
            self.set_parameter_requires_grad(feature_extracting=feature_extraction)

        if arch_type == "modified":
            # Remove last N=3 blocks: keeps blocks 0-5, output channels = 112
            N = 3
            self.backbone.features = nn.Sequential(
                *list(self.backbone.features.children())[:-N]
            )
            in_features = 112  # Output channels after removing last 3 blocks
        else:
            # Original architecture: 1280 features from EfficientNet-B0
            in_features = 1280

        # added for task 4 & 5: evaluate different methodologies to improve learning curve
        # Build classifier: [BatchNorm?] -> [Dropout?] -> Linear(output)
        layers = []
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(in_features))
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(in_features, num_classes))

        self.backbone.classifier = nn.Sequential(*layers)

    def forward(self, x):
        conv_feature_map = self.backbone.features(x)
        return self.backbone(x), conv_feature_map

    def extract_feature_maps(self, input_image: torch.Tensor):
        conv_weights = []
        conv_layers = []
        total_conv_layers = 0

        for module in self.backbone.features.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers += 1
                conv_weights.append(module.weight)
                conv_layers.append(module)

        print("TOTAL CONV LAYERS: ", total_conv_layers)
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        x = torch.clone(input=input_image)
        for layer in conv_layers:
            x = layer(x)
            feature_maps.append(x)
            layer_names.append(str(layer))

        return feature_maps, layer_names

    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output

            return hook

        # Register hooks for specified layers
        # for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.vgg16 = modify_fn(self.vgg16)

    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_grad_cam(
        self,
        input_image: torch.Tensor,
        target_layer: List[Type[nn.Module]],
        targets: List[Type[ClassifierOutputTarget]],
    ) -> Type[GradCAMPlusPlus]:
        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam


# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = TeacherModel(num_classes=8, feature_extraction=False)
    # model.load_state_dict(torch.load("saved_model.pt"))
    # model = model

    """
        features.0
        features.2
        features.5
        features.7
        features.10
        features.12
        features.14
        features.17
        features.19
        features.21
        features.24
        features.26
        features.28
    """

    transformation = F.Compose(
        [
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.RandomHorizontalFlip(p=1.0),
            F.Resize(size=(256, 256)),
        ]
    )
    # Example GradCAM usage
    dummy_input = Image.open(
        "/home/cboned/data/Master/MIT_split/test/highway/art803.jpg"
    )  # torch.randn(1, 3, 224, 224)
    input_image = transformation(dummy_input).unsqueeze(0)

    target_layers = [model.backbone.features[26]]
    targets = [ClassifierOutputTarget(6)]

    image = torch.from_numpy(np.array(dummy_input)).cpu().numpy()
    image = (
        (image - image.min()) / (image.max() - image.min())
    )  ## Image needs to be between 0 and 1 and be a numpy array (Remember that if you have norlized the image you need to denormalize it before applying this (image * std + mean))

    ## VIsualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.

    grad_cams = model.extract_grad_cam(
        input_image=input_image, target_layer=target_layers, targets=targets
    )

    visualization = show_cam_on_image(image, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

    # Display processed feature maps shapes
    feature_maps, layer_names = model.extract_feature_maps(input_image)

    ### Aggregate the feature maps
    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        min_feature_map, min_index = torch.min(
            feature_map, 0
        )  # Get the min across channels
        processed_feature_maps.append(min_feature_map.data.cpu().numpy())

    # Plot All the convolution feature maps separately
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i], cmap="hot", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)

    plt.show()

    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    with torch.no_grad():
        feature_map = (
            model.extract_features_from_hooks(x=input_image, layers=["features.28"])
        )["features.28"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        print(feature_map.shape)
        processed_feature_map, _ = torch.min(feature_map, 0)

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.show()

    ## Draw the model
    model_graph = draw_graph(
        model, input_size=(1, 3, 224, 224), device="meta", expand_nested=True, roll=True
    )
    model_graph.visual_graph.render(filename="test", format="png", directory="./Week3")
