from typing import Callable
from typing import List, Tuple

import numpy as np
import scipy.fftpack as fp
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageFilter import GaussianBlur, UnsharpMask
from kornia import gaussian_blur2d
from torch import Tensor
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor, to_pil_image

BLUR_ID = 100
SHARPEN_ID = 101
HPF_ID = 102
KHPF_ID = 103
LPF_ID = 104
TRANSFORMS = {'blur': BLUR_ID, 'sharpen': SHARPEN_ID, 'hpf': HPF_ID, 'lpf': LPF_ID}


class ConditionalCompose(Compose):
    def __init__(self, conditional_transforms: List[Tuple[int, Callable, Callable]], gpu=False):
        """
        This composes multiple torchvision transforms. However, each transformation has two versions.
        ConditionalCompose executes the first version if the label matches the condition and the other if not.
        Note that this class should not be used for data transformation during testing as it uses class labels!
        We used it to experiment with different version of frequency filters in our frequency analysis experiments, however,
        have only reported results for equal filters on all labels in the paper.
        @param conditional_transforms: A list of tuples (cond, trans1, trans2).
            ConditionalCompose iterates of all elements and at each time executes
            trans1 on the data if the label equals cond and trans2 if not cond.
        @param gpu: whether to move the data that is to be transformed to the gpu first.
        """
        super(ConditionalCompose, self).__init__(None)
        self.conditional_transforms = conditional_transforms
        self.gpu = gpu

    def __call__(self, img, tgt):
        for cond, t1, t2 in self.conditional_transforms:
            t1 = (lambda x: x) if t1 is None else t1
            t2 = (lambda x: x) if t2 is None else t2
            if not self.gpu:
                if tgt == cond:
                    img = t1(img)
                else:
                    img = t2(img)
            else:
                tgt = torch.Tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt
                tgt = tgt.to(img.device)
                img = torch.where(tgt.reshape(-1, 1, 1, 1) == cond, t1(img), t2(img))
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.conditional_transforms:
            format_string += '\n'
            format_string += F'    {repr(t[1])} if {repr(t[0])} else {repr(t[2])}'
        format_string += '\n)'
        return format_string


def get_transform(transform: int, magnitude: float) -> Callable:
    if transform == BLUR_ID:
        transform = CpuGaussianBlur(magnitude)
    elif transform == SHARPEN_ID:
        transform = PilUnsharpMask(magnitude)
    elif transform == HPF_ID:
        transform = DFTHighPassFilter(int(magnitude))
    elif transform == LPF_ID:
        transform = DFTLowPassFilter(int(magnitude))
    else:
        raise NotImplementedError()
    return transform


class PilGaussianBlur(object):
    def __init__(self, magnitude: float):
        """ applies a Gaussian blur to PIL images (cpu)"""
        self.magnitude = magnitude

    def __call__(self, x: Image):
        return x.filter(GaussianBlur(radius=self.magnitude))

    def __repr__(self):
        return f"PilGaussianBlur radius {self.magnitude}"


class CpuGaussianBlur(object):
    def __init__(self, magnitude: float):
        """ applies a Gaussian blur to PIL images using Kornia, for which it temporarily transforms the image to a tensor (CPU)"""
        self.magnitude = magnitude
        self.sigma = self.magnitude
        self.k = 2 * int(int(self.sigma / 2) + 0.5) + 1

    def __call__(self, img: Image) -> Tensor:
        if self.sigma <= 0:
            return img
        else:
            img = to_tensor(img).unsqueeze(0)
            k = max(min(self.k, 2 * int(int(img.size(-1) / 2) + 0.5) - 1), 3)
            img = gaussian_blur2d(img, (k, k), (self.sigma, self.sigma))
            img = to_pil_image(img.squeeze(0))
            return img

    def __repr__(self) -> str:
        return 'CPU-BLUR'

    def __str__(self) -> str:
        return 'CPU-BLUR'


class PilUnsharpMask(object):
    def __init__(self, magnitude: float):
        """ applies an UnsharpMask to PIL images (CPU)"""
        self.magnitude = magnitude

    def __call__(self, x: Image):
        return x.filter(UnsharpMask(percent=int(self.magnitude * 100)))

    def __repr__(self):
        return f"PilUnsharpMask percent {self.magnitude}"


class Normalize(object):
    def __init__(self, normalize: transforms.Normalize):
        """ applies a typical torchvision normalization to tensors (GPU compatible) """
        self.normalize = normalize

    def __call__(self, img: Tensor) -> Tensor:
        return self.normalize(img)

    def __repr__(self) -> str:
        return 'GPU-'+self.normalize.__repr__()

    def __str__(self) -> str:
        return 'GPU-'+self.normalize.__str__()


class Blur(object):
    def __init__(self, blur: PilGaussianBlur):
        """ applies a Gaussian blur to tensors using Kornia (GPU compatible), reuses the parameters of a given PilGaussianBlur """
        self.blur = blur
        self.sigma = blur.magnitude
        self.k = 2 * int(int(self.sigma / 2) + 0.5) + 1

    def __call__(self, img: Tensor) -> Tensor:
        if self.sigma <= 0:
            return img
        else:
            k = max(min(self.k, 2 * int(int(img.size(-1) / 2) + 0.5) - 1), 3)
            return gaussian_blur2d(img, (k, k), (self.sigma, self.sigma))

    def __repr__(self) -> str:
        return 'GPU-'+self.blur.__repr__()

    def __str__(self) -> str:
        return 'GPU-'+self.blur.__str__()


class ToGrayscale(object):
    def __init__(self, t: transforms.Grayscale):
        """ removes the color channels of a given tensor of images (GPU compatible) """
        pass

    def __call__(self, img: Tensor) -> Tensor:
        return img.mean(1).unsqueeze(1)

    def __repr__(self) -> str:
        return 'GPU-Grayscale'

    def __str__(self) -> str:
        return 'GPU-Grayscale'


class MinMaxNorm(object):
    def __init__(self, norm: 'MinMaxNorm' = None):
        """ applies an image-wise min-max normalization (brings to 0-1 range) to a tensor of images (GPU compatible) """
        pass

    def __call__(self, img: Tensor) -> Tensor:
        img = img.flatten(1).sub(img.flatten(1).min(1)[0].unsqueeze(1)).reshape(img.shape)
        img = img.flatten(1).div(img.flatten(1).max(1)[0].unsqueeze(1)).reshape(img.shape)
        return img

    def __repr__(self) -> str:
        return 'GPU-MinMaxNorm'

    def __str__(self) -> str:
        return 'GPU-MinMaxNorm'


class DFTHighPassFilter(object):
    def __init__(self, magnitude: int = 1):
        """ applies a true high pass filter to a PIL image using numpy and an FFT (CPU) """
        self.magnitude = magnitude

    def __call__(self, img: Image) -> Image:
        if self.magnitude <= 0:
            return img
        else:
            img = np.asarray(img).astype(float) / 255
            gray = len(img.shape) == 2
            if gray:
                img = img[:, :, None]
            h, w, c = img.shape
            n = min(self.magnitude, min(w // 2, h // 2))
            for cc in range(c):
                f1 = fp.fft2(img[:, :, cc])
                f2 = fp.fftshift(f1)
                f2[w//2-n:w//2+n, h//2-n:h//2+n] = 0
                img[:, :, cc] = fp.ifft2(fp.ifftshift(f2)).real
            img = img - img.min()
            img = img / img.max()
            img = (img * 255).astype(np.uint8)
            if gray:
                img = img[:, :, 0]
            return Image.fromarray(img)

    def __repr__(self) -> str:
        return f'DFT-HPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'DFT-HPF-{self.magnitude}'


class GpuDFTHighPassFilter(object):
    def __init__(self, hpf: DFTHighPassFilter):
        """
        Applies a true high pass filter to a tensor of images using torch and an FFT (GPU compatible).
        Reuses the params of a given CPU HPF.
        """
        self.magnitude = hpf.magnitude
        self.norm = MinMaxNorm()

    def __call__(self, img: Tensor) -> Tensor:
        if self.magnitude <= 0:
            return img
        else:
            n, c, h, w = img.shape
            e = min(self.magnitude, min(w // 2, h // 2))
            f1 = torch.fft.fft2(img)
            f2 = torch.fft.fftshift(f1)
            f2[:, :, h//2-e:h//2+e, w//2-e:w//2+e] = 0
            img = torch.fft.ifft2(torch.fft.ifftshift(f2)).real
            img = self.norm(img)
            return img

    def __repr__(self) -> str:
        return f'GPU-DFT-HPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'GPU-DFT-HPF-{self.magnitude}'


class DFTLowPassFilter(object):
    def __init__(self, magnitude: int = 1):
        """ applies a true low pass filter to a PIL image using numpy and an FFT (CPU) """
        self.magnitude = magnitude

    def __call__(self, img: Image) -> Image:
        if self.magnitude <= 0:
            return img
        else:
            img = np.asarray(img).astype(float) / 255
            gray = len(img.shape) == 2
            if gray:
                img = img[:, :, None]
            h, w, c = img.shape
            n = min(self.magnitude, min(w // 2, h // 2))
            for cc in range(c):
                f1 = fp.fft2(img[:, :, cc])
                f2 = fp.fftshift(f1)
                f2[:, :n, :] = 0
                f2[:, -n:, :] = 0
                f2[:, :, :n] = 0
                f2[:, :, -n:] = 0
                img[:, :, cc] = fp.ifft2(fp.ifftshift(f2)).real
            img = img - img.min()
            img = img / img.max()
            img = (img * 255).astype(np.uint8)
            if gray:
                img = img[:, :, 0]
            return Image.fromarray(img)

    def __repr__(self) -> str:
        return f'DFT-LPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'DFT-LPF-{self.magnitude}'


class GpuDFTLowPassFilter(object):
    def __init__(self, lpf: DFTLowPassFilter):
        """
        Applies a true low pass filter to a tensor of images using torch and an FFT (GPU compatible).
        Reuses the params of a given CPU LPF.
        """
        self.magnitude = lpf.magnitude
        self.norm = MinMaxNorm()

    def __call__(self, img: Tensor) -> Tensor:
        if self.magnitude <= 0:
            return img
        else:
            n, c, h, w = img.shape
            e = min(self.magnitude, min(w // 2, h // 2))
            f1 = torch.fft.fft2(img)
            f2 = torch.fft.fftshift(f1)
            f2[:, :, :e, :] = 0
            f2[:, :, -e:, :] = 0
            f2[:, :, :, :e] = 0
            f2[:, :, :, -e:] = 0
            img = torch.fft.ifft2(torch.fft.ifftshift(f2)).real
            img = self.norm(img)
            return img

    def __repr__(self) -> str:
        return f'GPU-DFT-LPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'GPU-DFT-LPF-{self.magnitude}'


class GlobalContrastNormalization(object):
    def __init__(self, gcn=None, scale='l1'):
        """
        Applies a global contrast normalization to a tensor of images;
        i.e. subtract mean across features (pixels) and normalize by scale,
        which is either the standard deviation, L1- or L2-norm across features (pixels).
        Note this is a *per sample* normalization globally across features (and not across the dataset).
        This is GPU compatible.
        """
        self.scale = scale
        if gcn is not None:
            assert gcn.scale == scale

    def __call__(self, x: torch.Tensor):  # x in [n, c, h, w]
        assert self.scale in ('l1', 'l2')
        n_features = int(np.prod(x.shape[1:]))
        mean = torch.mean(x.flatten(1), dim=1)[:, None, None, None]  # mean over all features (pixels) per sample
        x -= mean
        if self.scale == 'l1':
            x_scale = torch.mean(torch.abs(x.flatten(1)), dim=1)[:, None, None, None]
        if self.scale == 'l2':
            x_scale = torch.sqrt(torch.sum(x.flatten(1) ** 2, dim=1))[:, None, None, None] / n_features
        x /= x_scale
        return x


GPU_TRANSFORMS = {  # maps CPU versions of transformations to corresponding GPU versions
    transforms.Normalize: Normalize, CpuGaussianBlur: Blur, type(None): lambda x: None,
    Normalize: Normalize, Blur: Blur, MinMaxNorm: MinMaxNorm,
    DFTHighPassFilter: GpuDFTHighPassFilter,
    GlobalContrastNormalization: GlobalContrastNormalization,   # there is no CPU version implemented so far
    DFTLowPassFilter: GpuDFTLowPassFilter
}
