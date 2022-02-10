
from torch import nn
from .filter_module import AddGaussianNoise





class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=None, k_size=3, sigma=0.1, lowpass_k_size=5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # self.smoothing = GaussianSmoothing(channel, lowpass_k_size, sigma)
        # self.addnoise = AddGaussianNoise(0,sigma)



    def forward(self, x):

        # x_pad = nn.functional.pad(x, (2, 2, 2, 2), mode='reflect')

        # x_noise = self.addnoise(x)

        # x_lowpass = self.smoothing(x_pad)

        # x_highpass = x - x_lowpass




        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        # y = self.addnoise(y)



        return x * y.expand_as(x)