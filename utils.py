import torch, math
import torch.nn.functional as F
import numpy as np

# reference from https://github.com/jianzhangcs/ISTA-Net-PyTorch
def my_zero_pad(img, block_size=32):
    old_h, old_w = img.shape
    delta_h = (block_size - np.mod(old_h, block_size)) % block_size
    delta_w = (block_size - np.mod(old_w, block_size)) % block_size
    img_pad = np.concatenate((img, np.zeros([old_h, delta_w])), axis=1)
    img_pad = np.concatenate((img_pad, np.zeros([delta_h, old_w + delta_w])), axis=0)
    new_h, new_w = img_pad.shape
    return img, old_h, old_w, img_pad, new_h, new_w

# reference from https://github.com/cszn
def H(img, mode, inv=False):
    if inv:
        mode = [0, 1, 2, 5, 4, 3, 6, 7][mode]
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])
"""
def SolveFFT(X, D, Y, alpha, x_size):
    '''
        X: N, 1, C_in, H, W, 2
        D: N, C_out, C_in, H, W, 2
        Y: N, C_out, 1, H, W, 2
        alpha: N, 1, 1, 1
    '''
    alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

    _D = cconj(D)
    Z = cmul(Y, D) + alpha * X

    factor1 = Z / alpha

    numerator = cmul(_D, Z).sum(2, keepdim=True)
    denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                        alpha.squeeze(-1)**2)
    factor2 = cmul(D, cdiv(numerator, denominator))
    X = (factor1 - factor2).mean(1)
    return torch.fft.irfft2(torch.view_as_complex(X), s=tuple(x_size))
"""
def SolveFFT(X, D, Y, alpha, x_size):
    '''
        X: N, 1, C_in, H, W, 2
        D: N, C_out, C_in, H, W, 2
        Y: N, C_out, 1, H, W, 2
        alpha: N, 1, 1, 1
    '''
    alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

    _D = cconj(D)
    Z = cmul(Y, D) + alpha * X

    factor1 = Z / alpha

    numerator = cmul(_D, Z).sum(2, keepdim=True)
    denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                        alpha.squeeze(-1)**2)
    factor2 = cmul(D, cdiv(numerator, denominator))
    X = (factor1 - factor2).mean(1)
    return torch.fft.irfft2(torch.view_as_complex(X), s=tuple(x_size))

def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)

def csum(x, y):
    # complex + real
    real = x[..., 0] + y
    img = x[..., 1]
    return torch.stack([real, img.expand_as(real)], -1)

def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)

def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c

def roll(psf, kernel_size, reverse=False):
    for axis, axis_size in zip([-2, -1], kernel_size):
        psf = torch.roll(psf,
                         int(axis_size / 2) * (-1 if not reverse else 1),
                         dims=axis)
    return psf

def p2o(psf, shape):
    """
    Point-spread function -> Optical transfer function (FFT).
    psf: [N, C, h, w]  空间域卷积核
    shape: [H, W]      目标图像尺寸
    return: [N, C, H, W//2+1, 2]  实部+虚部
    """
    kernel_size = (psf.size(-2), psf.size(-1))
    # 1. 零填充到目标图像大小
    pad_h = shape[0] - kernel_size[0]
    pad_w = shape[1] - kernel_size[1]
    psf = F.pad(psf, [0, pad_w, 0, pad_h])          # 右、下填充

    # 2. 循环移位，中心->左上角
    psf = roll(psf, kernel_size)

    # 3. 2D RFFT → 只保留正频率
    otf = torch.fft.rfft2(psf, dim=(-2, -1))        # 复数

    # 4. 拆成实部+虚部 → 5维
    otf = torch.view_as_real(otf)                   # [..., H, W//2+1, 2]
    return otf