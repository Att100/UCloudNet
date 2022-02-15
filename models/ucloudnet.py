import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BasicConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1):
        super().__init__()

        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = F.relu6(out)
        return self.bn(out)

class DoubleConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = BasicConv2d(in_channels, out_channels, 3, 1)
        self.conv2 = BasicConv2d(out_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.in_channels != self.out_channels:
            return out
        else:
            return out + x

class UNetDown(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2D(3, 2, 1)
        self.conv = BasicConv2d(in_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        return out
        
class UNetUp(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BasicConv2d(in_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class UCloudNet(nn.Layer):
    def __init__(self, in_channels=1, base_channels=64, n_classes=2, aux=True):
        super().__init__()

        self.out_channels = n_classes if n_classes != 2 else 1 
        self.aux = aux

        # in 
        self.in_conv = BasicConv2d(in_channels, base_channels, 3, 1)

        # encoder
        self.double_conv_1 = DoubleConv2d(base_channels, base_channels)
        self.down_1 = UNetDown(base_channels, base_channels*2)
        self.double_conv_2 = DoubleConv2d(base_channels*2, base_channels*2)
        self.down_2 = UNetDown(base_channels*2, base_channels*4)
        self.double_conv_3 = DoubleConv2d(base_channels*4, base_channels*4)
        self.down_3 = UNetDown(base_channels*4, base_channels*8)
        self.double_conv_4 = DoubleConv2d(base_channels*8, base_channels*8)
        self.down_4 = UNetDown(base_channels*8, base_channels*16)
        
        self.double_conv_5 = DoubleConv2d(base_channels*16, base_channels*16)

        # decoder
        self.up_1 = UNetUp(base_channels*16, base_channels*8)
        self.double_conv_6 = DoubleConv2d(base_channels*16, base_channels*8)
        self.up_2 = UNetUp(base_channels*8, base_channels*4)
        self.double_conv_7 = DoubleConv2d(base_channels*8, base_channels*4)
        self.up_3 = UNetUp(base_channels*4, base_channels*2)
        self.double_conv_8 = DoubleConv2d(base_channels*4, base_channels*2)
        self.up_4 = UNetUp(base_channels*2, base_channels)
        self.double_conv_9 = DoubleConv2d(base_channels*2, base_channels)

        # classifier
        self.dp = nn.Dropout2D(p=0.2)
        self.classifier = nn.Conv2D(base_channels, self.out_channels, 3, 1, 1)

        # aux classifier
        if self.aux:
            self.aux_x2 = BasicConv2d(base_channels*2, self.out_channels, 1)
            self.aux_x4 = BasicConv2d(base_channels*4, self.out_channels, 1)

    def forward(self, x):
        out = self.in_conv(x)
        feat_1 = self.double_conv_1(out)
        out = self.down_1(feat_1)
        feat_2 = self.double_conv_2(out)
        out = self.down_2(feat_2)
        feat_3 = self.double_conv_3(out)
        out = self.down_3(feat_3)
        feat_4 = self.double_conv_4(out)
        out = self.down_4(feat_4)

        out = self.double_conv_5(out)

        out = self.up_1(out)
        out = paddle.concat([out, feat_4], axis=1)
        out = self.double_conv_6(out)
        out = self.up_2(out)
        out = paddle.concat([out, feat_3], axis=1)
        up_x4 = self.double_conv_7(out)
        out = self.up_3(up_x4)
        out = paddle.concat([out, feat_2], axis=1)
        up_x2 = self.double_conv_8(out)
        out = self.up_4(up_x2)
        out = paddle.concat([out, feat_1], axis=1)
        out = self.double_conv_9(out)

        out = self.classifier(self.dp(out))
        if self.aux:
            up_x2_out = self.aux_x2(up_x2)
            up_x4_out = self.aux_x4(up_x4)
            return out, up_x2_out, up_x4_out
        else:
            return out

def _bce_loss_with_aux(pred, target, weight=[1, 0.4, 0.2]):
    # pred = (x1, x2, x4)
    pred, pred_sub2, pred_sub4 = tuple(pred)

    target_2x = F.interpolate(
        target, pred_sub2.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, pred_sub4.shape[2:], mode='bilinear', align_corners=True)
    
    _1x_loss = F.binary_cross_entropy(F.sigmoid(pred), target)
    _2x_loss = F.binary_cross_entropy(F.sigmoid(pred_sub2), target_2x)
    _4x_loss = F.binary_cross_entropy(F.sigmoid(pred_sub4), target_4x)

    loss = weight[0] * _1x_loss + weight[1] * _2x_loss + weight[2] * _4x_loss
    return loss, (_1x_loss, _2x_loss, _4x_loss)