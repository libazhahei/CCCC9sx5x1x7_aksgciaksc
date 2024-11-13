import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, out_channels, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(out_channels)
        self.pool2 = nn.AdaptiveAvgPool2d(out_channels)
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
    

class CustomizedDeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError(
                "Output stride should be 8 or 16, got {}.".format(output_stride)
            )

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(ConvBN, self).__init__()
        if in_channels == out_channels:
            groups = in_channels  # Use depthwise convolution if input and output channels are the same
        else:
            groups = 1  # Use regular convolution otherwise
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)  # Apply convolution and batch normalization
        return out


class MultiResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 0) -> None:
        super(MultiResidualBlock, self).__init__()
        self.conv1_0 = ConvBN(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)  # 3x3 convolution path
        self.conv1_1 = ConvBN(in_channels, out_channels, kernel_size=1, padding=dilation, dilation=dilation)  # 1x1 convolution path
        self.conv1_2 = ConvBN(in_channels, out_channels, kernel_size=1, padding=dilation, dilation=dilation)  # Another 1x1 convolution path

        self.relu = nn.ReLU()  # Activation function

        self.conv2 = ConvBN(out_channels, out_channels, kernel_size=1, padding=0)  # 1x1 convolution

        self.conv3_0 = ConvBN(out_channels, out_channels, kernel_size=3, padding=1)  # 3x3 convolution
        self.conv3_1 = ConvBN(out_channels, out_channels, kernel_size=1, padding=0)  # 1x1 convolution
        # self.conv3_2 = ConvBN(out_channels, out_channels, kernel_size=1, padding=0)  # 1x1 convolution

    def forward(self, x):
        x1_0 = self.conv1_0(x)  # Apply 3x3 convolution
        x1_1 = self.conv1_1(x)  # Apply 1x1 convolution
        x1_2 = self.conv1_2(x)  # Apply another 1x1 convolution

        x1 = self.relu(x1_0 + x1_1)  # Sum and activate
        x2 = self.relu(self.conv2(x1) + x1)  # Apply 1x1 conv, sum with x1, and activate

        x3 = self.relu(self.conv3_0(x2) + self.conv3_1(x2))  # Apply 3x3 and 1x1 convs, sum and activate
        out = x3 + x1_2  # Final summation
        return out
    
class CustimizedASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            # nn.Conv2d(
            #     in_channels,
            #     out_channels,
            #     kernel_size=3,
            #     padding=dilation,
            #     dilation=dilation,
            #     bias=False,
            # ),
            MultiResidualBlock(
                in_channels,
                out_channels,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class CustomizedASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            # nn.AdaptiveAvgPool2d(1),
            StripPooling(in_channels, out_channels, {"mode": "bilinear", "align_corners": False}),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # super().__init__(
        #     StripPooling(in_channels, out_channels, {"mode": "bilinear", "align_corners": False}),
        # )

    def forward(self, x):
        size = x.shape[-2:]
        for i,mod in enumerate(self):
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = CustimizedASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(CustomizedASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(dephtwise_conv, pointwise_conv)

class CustomizedDeeplabv3plus(smp.DeepLabV3Plus):
    def __init__(self, encoder_name, encoder_weights, classes, activation):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes, activation=activation)
        self.decoder = CustomizedDeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12,24,36),
            output_stride=16,
        )

# DEEPLAB_PATH = "/kaggle/input/mydeeplabv3p/pytorch/default/2/customized_deeplabv3_best_model.pth"
# create segmentation model with pretrained encoder
# customized_deeplabv3_all = load_model(None, CustomizedDeeplabv3plus(
#     encoder_name='resnet101', 
#     encoder_weights='imagenet', 
#     classes=4, 
#     activation='softmax2d',
# ))

def get_model(checkpoint_path=None, num_classes=4):
    model = CustomizedDeeplabv3plus(
        encoder_name='resnet101', 
        encoder_weights='imagenet', 
        classes=num_classes, 
        activation='softmax2d',
    )
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return model