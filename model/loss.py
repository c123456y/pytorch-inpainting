import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from model.layer import VGG19FeatLayer
from torch.nn.functional import conv2d
import numpy as np


class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -input.mean()
        return {'g_loss': g_loss, 'd_loss': d_loss}


def gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(1, real_data.nelement()).contiguous().view(1, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


def random_interpolate(gt, pred):
    batch_size = gt.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    interpolated = gt * alpha + pred * (1 - alpha)
    return interpolated


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_x, w_x = x.size()[2:]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        loss = torch.sum(h_tv) + torch.sum(w_tv)
        return loss


import torchvision.models as models


class VGG19(nn.Module):
    def __init__(self, resize_input=False):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.resize_input = resize_input
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        names = list(zip(prefix, posfix))
        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(
                pre, pos), torch.nn.Sequential())

        nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8],
                [9, 10, 11], [12, 13], [14, 15], [16, 17],
                [18, 19, 20], [21, 22], [23, 24], [25, 26],
                [27, 28, 29], [30, 31], [32, 33], [34, 35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # resize and normalize input for pretrained vgg19
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        if self.resize_input:
            x = F.interpolate(
                x, size=(256, 256), mode='bilinear', align_corners=True)
        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)
        out = {key: value for (key, value) in list(zip(self.relus, features))}
        return out


class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss


class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss


class advloss():
    def __init__(self, alpha=1.0, beta=0.5):
        self.celoss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, netD, fake, real):
        fake_detach = fake.detach()
        d_fake = netD(fake_detach)
        d_real = netD(real)
        dis_loss = self.celoss(-d_real).mean() + self.celoss(d_fake).mean()

        g_fake = netD(fake)
        g_real = netD(real)
        gen_loss = self.loss_fn(-g_fake).mean()
        gen_loss = self.alpha * self.celoss(-g_fake, -g_real) - self.beta * self.celoss.forward(-g_fake, -g_real)

        return dis_loss, gen_loss


class ganloss():
    def __init__(self, ksize=71):
        self.ksize = ksize
        self.loss_fn = nn.MSELoss(reduction='mean')

    def __call__(self, netD, fake, real, masks):
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.size()
        b, c, ht, wt = masks.size()

        # Handle inconsistent size between outputs and masks
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)

        d_fake_label = gaussian_blur(masks, (self.ksize, self.ksize), (10, 10)).detach().cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label) * masks / torch.mean(masks)

        return dis_loss.mean(), gen_loss.mean()


def focal_loss(netD, fake, mask, ratio, gamma=2):
    fake_detach = fake.detach()

    d_fake = netD(fake_detach)

    _, _, h, w = d_fake.size()
    b, c, ht, wt = mask.size()

    # Handle inconsistent size between outputs and masks
    if h != ht or w != wt:
        pred = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)

    y_pred = nn.softmax(pred, axis=-1)
    pred1, pred2 = torch.split(y_pred, [1, 1], axis=-1)
    mask = torch.float(mask)

    log1 = torch.mul(pred2 ** gamma, torch.log(np.clip(pred1, 1e-8, 1.0 - 1e-8)))
    log2 = torch.mul(pred1 ** gamma, torch.log(np.clip(pred2, 1e-8, 1.0 - 1e-8)))

    loss = - ratio * torch.mul(1 - mask, log1).permute(1, 2, 3, 0) - \
           (1 - ratio) * torch.mul(mask, log2).permute(1, 2, 3, 0)

    loss = loss.permute(3, 0, 1, 2)
    loss = torch.mean(loss)

    return loss


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x)))
                         for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.
    Args:
      kernel_size (int): filter size. It should be odd and positive.
      sigma (float): gaussian standard deviation.
    Returns:
      Tensor: 1D tensor with gaussian filter coefficients.
    Shape:
      - Output: :math:`(\text{kernel_size})`

    Examples::
      >>> kornia.image.get_gaussian_kernel(3, 2.5)
      tensor([0.3243, 0.3513, 0.3243])
      >>> kornia.image.get_gaussian_kernel(5, 1.5)
      tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(
            "kernel_size must be an odd positive integer. Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size, sigma):
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
      kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
        Sizes should be odd and positive.
      sigma (Tuple[int, int]): gaussian standard deviation in the x and y
        direction.
    Returns:
      Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
      - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::
      >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
      tensor([[0.0947, 0.1183, 0.0947],
              [0.1183, 0.1478, 0.1183],
              [0.0947, 0.1183, 0.0947]])

      >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
      tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
              [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
              [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class GaussianBlur(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.
    Arguments:
      kernel_size (Tuple[int, int]): the size of the kernel.
      sigma (Tuple[float, float]): the standard deviation of the kernel.
    Returns:
      Tensor: the blurred tensor.
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples::
      >>> input = torch.rand(2, 4, 5, 5)
      >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
      >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._padding = self.compute_zero_padding(kernel_size)
        self.kernel = get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError(
                "Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # TODO: explore solution when using jit.trace since it raises a warning
        # because the shape is converted to a tensor instead to a int.
        # convolve tensor with gaussian kernel
        return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


######################
# functional interface
######################

def gaussian_blur(input, kernel_size, sigma):
    r"""Function that blurs a tensor using a Gaussian filter.
    See :class:`~kornia.filters.GaussianBlur` for details.
    """
    return GaussianBlur(kernel_size, sigma)(input)





