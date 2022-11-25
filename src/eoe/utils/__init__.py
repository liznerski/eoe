import cv2
import torch
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt


def imshow(tensors: torch.Tensor, ms=100, name='out', nrow=8, norm=True, use_plt=True, save: str = None, show: bool = True):
    """
    Given a tensor of images, this immediately displays them as a matrix of images using either matplotlib or opencv.
    @param tensors: tensor of images nxcxhxw
    @param ms: milliseconds to block while showing, only works for opencv.
    @param name: name of the window that displays the images.
    @param nrow: the number of images shown per row.
    @param norm: whether to normalize the images to 0-1 range.
    @param use_plt: whether to use matplotlib or opencv.
    @param save: a path where the image matrix is also saved to.
    @param show: whether to actually show the image.
    """
    if use_plt:
        matplotlib.use('TkAgg' if show else 'Agg')
    if isinstance(tensors, (list, tuple)):
        assert len(set([t.dim() for t in tensors])) == 1 and tensors[0].dim() == 4
        tensors = [t.float().div(255) if t.dtype == torch.uint8 else t for t in tensors]
        tensors = [t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t for t in tensors]
        tensors = torch.cat(tensors)
    if tensors.dtype == torch.uint8:
        tensors = tensors.float().div(255)
    t = vutils.make_grid(tensors, nrow=nrow, scale_each=norm)
    t = t.transpose(0, 2).transpose(0, 1).numpy()
    if use_plt:
        plt.close()
        if norm:
            plt.imshow(t, resample=True)
        else:
            plt.imshow(t, resample=True, vmin=0, vmax=1)
        if save is not None:
            plt.imsave(save, t)
        if show:
            plt.show()
            plt.pause(0.001)
    else:
        if t.shape[-1] == 3:
            t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, 1280, 768)
        if save is not None:
            cv2.imwrite(save, t)
        if show:
            cv2.imshow(name, t)
            cv2.waitKey(ms)

