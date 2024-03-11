# translation from https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/color/colorconv.py#L620
import math
import torch
from torch import Tensor

rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]]).to(torch.float32)  #M: SCA matrix
hed_from_rgb = torch.linalg.inv(rgb_from_hed) #D: color deconvolution matrix

def separate_stains(rgb: Tensor, conv_matrix: Tensor, eps=1e-5, ddtype=torch.float32)->Tensor:
    """RGB to stain color space conversion.
    """
    if rgb.dim()==3:
        rgb = rgb.unsqueeze(0)
    dtype = rgb.dtype
    device = rgb.device
    rgb = rgb.to(ddtype)
    log_adjust = math.log(eps)  # used to compensate the sum above
    #stains = (torch.log(torch.where(rgb>eps, rgb, eps)) / log_adjust) @ conv_matrix.to(device, ddtype)
    stains = torch.einsum('bchw,ck->bkhw', (torch.log(torch.where(rgb>eps, rgb, eps * torch.ones_like(rgb))) / log_adjust), conv_matrix.to(device, ddtype))
    return torch.where(stains>0.0, stains, torch.zeros_like(stains)).to(dtype)

def combine_stains(stains: Tensor, conv_matrix: Tensor, eps=1e-5, ddtype=torch.float32)->Tensor:
    """Stain to RGB color space conversion.
    """
    if stains.dim()==3:
        stains = stains.unsqueeze(0)
    dtype = stains.dtype
    device = stains.device
    stains = stains.to(ddtype)
    log_adjust = -math.log(eps)
    #log_rgb = -(stains * log_adjust) @ conv_matrix.to(device, ddtype)
    log_rgb = torch.einsum('bchw,ck->bkhw', -(stains * log_adjust), conv_matrix.to(device, ddtype))
    rgb = torch.exp(log_rgb)

    return torch.clamp(rgb, 0, 1).to(dtype)

def rgb2hed(rgb:Tensor, ddtype=torch.float32)->Tensor:
    """RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.

    Parameters
    ----------
    rgb : (B, C, H, W) Tensor
        The image in RGB format.

    Returns
    -------
    out : (B, C, H, W) Tensor
        The image in HED format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not the right shape.

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> ihc_hed = rgb2hed(ihc)
    """
    return separate_stains(rgb, hed_from_rgb, ddtype=ddtype)


def hed2rgb(hed:Tensor, ddtype=torch.float32)->Tensor:
    """Haematoxylin-Eosin-DAB (HED) to RGB color space conversion.
    Parameters
    ----------
    rgb : (B, C, H, W) Tensor
        The image in HED format.

    Returns
    -------
    out : (B, C, H, W) Tensor
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not the right shape.
    """
    return combine_stains(hed, rgb_from_hed, ddtype=ddtype)