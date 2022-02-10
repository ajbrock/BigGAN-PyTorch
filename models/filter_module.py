import torch
from torch import nn
import numbers
import math
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, _sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(_sigma, numbers.Number):
            sigma = [_sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.myconv = nn.functional.conv1d
        elif dim == 2:
            self.myconv = nn.functional.conv2d
        elif dim == 3:
            self.myconv = nn.functional.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.myconv(input, weight=self.weight, groups=self.groups)






class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):

        return x + torch.cuda.FloatTensor(x.size()).normal_(self.mean, self.std)


if __name__ == '__main__':

    TEST_IMG = "../data/IAPS/1019.jpg"
    pil_img = Image.open(TEST_IMG)
    print(pil_img.size)

    pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)
    print(pil_to_tensor.shape)

    pil_to_tensor_pad = F.pad(pil_to_tensor, (2, 2, 2, 2), mode='reflect')
    smoothing = GaussianSmoothing(3, 21, 21)
    output_lowpass_tensor = smoothing(pil_to_tensor_pad)
    output_lowpass = output_lowpass_tensor.squeeze(0)
    output_lowpass = output_lowpass.permute(1, 2, 0)

    plt.imshow(output_lowpass)
    plt.show()
    #
    # output_highpass = pil_to_tensor - output_lowpass_tensor
    # output_highpass = output_highpass.squeeze(0)
    # output_highpass = output_highpass.permute(1, 2, 0)

    # plt.imshow(output_highpass)
    # plt.show()

