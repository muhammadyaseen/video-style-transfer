# Same transform net, but implemented with Depthwise Separable Convolutions

# this class implements the DwConv block introduced in MobileNet v1 paper
import torch

class MobileNetConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=0):

        super(MobileNetConvBlock, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, stride=stride, kernel_size=kernel_size, padding=pad, groups=in_channels)
        #self.mnb_bn1 = torch.nn.BatchNorm2d(in_channels)
        self.mnb_relu1 = torch.nn.ReLU()
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1)
        #self.mnb_bn2 = torch.nn.BatchNorm2d(out_channels)
        self.mnb_relu2 = torch.nn.ReLU()

    def forward(self, x):

        out = self.depthwise(x)
        #out = self.mnb_bn1(out)
        out = self.mnb_relu1(out)
        out = self.pointwise(out)
        #out = self.mnb_bn2(out)
        out = self.mnb_relu2(out)

        return out

class DSC_TransformerNet(torch.nn.Module):

    def __init__(self):

        super(DSC_TransformerNet, self).__init__()

        # Initial convolution layers
        self.conv1 = DSC_ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = DSC_ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = DSC_ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res1 = DSC_ResidualBlock(128)
        self.res2 = DSC_ResidualBlock(128)
        self.res3 = DSC_ResidualBlock(128)
        self.res4 = DSC_ResidualBlock(128)
        self.res5 = DSC_ResidualBlock(128)

        # Upsampling Layers
        self.deconv1 = DSC_UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = DSC_UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)

        self.deconv3 = DSC_ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):

        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class DSC_ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(DSC_ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = MobileNetConvBlock(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class DSC_ResidualBlock(torch.nn.Module):

    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):

        super(DSC_ResidualBlock, self).__init__()
        self.conv1 = DSC_ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = DSC_ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class DSC_UpsampleConvLayer(torch.nn.Module):

    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):

        super(DSC_UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = MobileNetConvBlock(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
