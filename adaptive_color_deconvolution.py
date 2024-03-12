import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler

# translation from https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/color/colorconv.py#L620
# Define matrices for RGB to Haematoxylin-Eosin-DAB (HED) color space conversion
rgb_from_hed: Tensor = torch.tensor([[0.65, 0.70, 0.29],
                                     [0.07, 0.99, 0.11],
                                     [0.27, 0.57, 0.78]], dtype=torch.float32)
hed_from_rgb: Tensor = torch.linalg.inv(rgb_from_hed)

def separate_stains(rgb: Tensor, conv_matrix: Tensor, eps: float = 1e-5, ddtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert an RGB image to stain color space using a specified conversion matrix.

    Parameters:
    - rgb: Tensor of shape (B, C, H, W) representing the image in RGB format.
    - conv_matrix: Tensor representing the color deconvolution matrix for conversion.
    - eps: Small value to prevent division by zero and logarithm of zero.
    - ddtype: Data type for the computation.

    Returns:
    - Tensor of the image in the stain color space.
    """
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    if conv_matrix.dim() == 2:
        conv_matrix = conv_matrix.unsqueeze(0)
    dtype = rgb.dtype
    device = rgb.device
    rgb = rgb.to(ddtype)
    log_adjust = math.log(eps)
    stains = torch.einsum('bchw,bck->bkhw', (torch.log(torch.where(rgb > eps, rgb, eps * torch.ones_like(rgb))) / log_adjust), conv_matrix.to(device, ddtype))
    return torch.where(stains > 0.0, stains, torch.zeros_like(stains)).to(dtype)

def combine_stains(stains: Tensor, conv_matrix: Tensor, eps: float = 1e-5, ddtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert a stain color space image back to RGB using a specified conversion matrix.

    Parameters:
    - stains: Tensor of shape (B, C, H, W) representing the image in stain color space.
    - conv_matrix: Tensor representing the color convolution matrix for conversion.
    - eps: Small value to prevent division by zero.
    - ddtype: Data type for the computation.

    Returns:
    - Tensor of the image in RGB format.
    """
    if stains.dim() == 3:
        stains = stains.unsqueeze(0)
    if conv_matrix.dim() == 2:
        conv_matrix = conv_matrix.unsqueeze(0)
    dtype = stains.dtype
    device = stains.device
    stains = stains.to(ddtype)
    log_adjust = -math.log(eps)
    log_rgb = torch.einsum('bchw,bck->bkhw', -(stains * log_adjust), conv_matrix.to(device, ddtype))
    rgb = torch.exp(log_rgb)
    return torch.clamp(rgb, 0, 1).to(dtype)

def rgb2hed(rgb: Tensor, ddtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert an RGB image to HED (Haematoxylin-Eosin-DAB) color space.

    Parameters:
    - rgb: Tensor of shape (B, C, H, W) representing the image in RGB format.
    - ddtype: Data type for the computation.

    Returns:
    - Tensor of the image in HED color space.
    """
    return separate_stains(rgb, hed_from_rgb, ddtype=ddtype)

def hed2rgb(hed: Tensor, ddtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert an HED (Haematoxylin-Eosin-DAB) color space image back to RGB.

    Parameters:
    - hed: Tensor of shape (B, C, H, W) representing the image in HED color space.
    - ddtype: Data type for the computation.

    Returns:
    - Tensor of the image in RGB format.
    """
    return combine_stains(hed, rgb_from_hed, ddtype=ddtype)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler

def calculate_sca_matrix(alphas: Tensor, betas: Tensor) -> Tensor:
    """
    Calculate the stain color appearance (SCA) matrix based on alpha and beta angles.

    Parameters:
    - alphas: Tensor representing the angle for the computation of the sine component.
    - betas: Tensor representing the angle for the computation of the cosine component.

    Returns:
    - Tensor representing the SCA matrix.
    """
    M = torch.stack([torch.cos(alphas) * torch.sin(betas), 
                     torch.cos(alphas) * torch.cos(betas),
                     torch.sin(alphas)], dim=1).permute(0, 2, 1).contiguous()
    return M

def loss_fn(hed: Tensor, lambda_p: float = 0.002, lambda_b: float = 10.0, lambda_e: float = 1.0, gamma: float = 0.3, eta: float = 0.6, eps: float = 1e-5, mask: Tensor = None) -> Tensor:
    """
    Compute the loss for the adaptive color deconvolution based on HED space.

    Parameters:
    - hed: Tensor of the image in HED color space.
    - lambda_p, lambda_b, lambda_e, gamma, eta: Parameters for the loss computation.
    - eps: Small value to prevent division by zero.
    - mask: Optional tensor to apply a mask to the HED image.

    Returns:
    - Computed loss value as a Tensor.
    """
    
    if mask is not None:
        assert hed.size() == mask.size()
        hed *= mask
    h, e, d = hed[:, 0], hed[:, 1], hed[:, 2]
    
    lp = d**2 + lambda_p * 2 * (h*e)/(h**2 + e**2 + eps)
    lp_mean = lp.mean()
    
    lb = (1-eta) * h.mean() - eta * e.mean()
    lb_sq = lb**2
    
    le = gamma - (h+e).mean()
    le_sq = le**2
    
    loss = lp_mean + lambda_b * lb_sq + lambda_e * le_sq
    return loss

def acd(img_tensor: Tensor, device: torch.device, lr: float = 0.001, itr: int = 300, enable_autocast: bool = True) -> tuple:
    """
    Perform adaptive color deconvolution on a batch of image tensors.

    Parameters:
    - img_tensor: Tensor of the images in RGB format.
    - device: The torch device to perform computations on.
    - lr: Learning rate for the optimizer.
    - itr: Number of iterations for optimization.
    - enable_autocast: Flag to enable/disable autocasting for mixed precision.

    Returns:
    - Tuple of matrices M (SCA matrix), D (deconvolution matrix), and W (weight matrix) for the batch.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    rgb_from_hed_local = rgb_from_hed.to(device)
    B = img_tensor.size(0)
    
    alphas = torch.asin(rgb_from_hed_local[:, 2])
    betas = torch.asin(rgb_from_hed_local[:, 0]/torch.cos(alphas))
    alphas = torch.nn.Parameter(alphas.unsqueeze(0).repeat(B, 1).to(device))
    betas = torch.nn.Parameter(betas.unsqueeze(0).repeat(B, 1).to(device))
    
    _W = torch.nn.Parameter(torch.eye(3)[:,:2].unsqueeze(0).repeat(B, 1, 1).to(device))
   
    
    optimizer = optim.Adagrad([_W, alphas, betas], lr=lr)
    scaler = GradScaler()
    
    for i in range(itr):
        optimizer.zero_grad()
        M = calculate_sca_matrix(alphas, betas)
        D = torch.linalg.inv(M)
        W = torch.cat([_W, torch.eye(3)[2].view(1, 3, 1).repeat(B, 1, 1).to(device)], dim=2)
        with autocast(enabled=enable_autocast):
            hed = separate_stains(img_tensor, W @ D)
            loss = loss_fn(hed).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    with torch.no_grad():
        M = calculate_sca_matrix(alphas, betas)
        D = torch.linalg.inv(M)
        W = torch.cat([_W, torch.eye(3)[2].view(1, 3, 1).repeat(B, 1, 1).to(device)], dim=2)
    return M.data.clone().detach(), D.data.clone().detach(), W.data.clone().detach()

# helper function to calculate NMI metrics
@torch.no_grad()
def calculate_normalized_median_intensity(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the normalized median intensity (NMI) of an image tensor.

    The NMI is defined as the ratio of the median intensity to the 95th percentile intensity
    of the image, computed across all channels. This function operates on batches of images
    and can handle both single and multiple images. The input tensor is expected to be in the
    format (B, C, H, W) for batches or (C, H, W) for single images.

    Parameters:
    - img_tensor (torch.Tensor): The input image tensor. The tensor should have dimensions
      (B, C, H, W) for batches of images or (C, H, W) for single images, where B is the batch size,
      C is the number of channels, H is the height, and W is the width of the images.

    Returns:
    - torch.Tensor: A tensor containing the normalized median intensity for each image in the batch.
      The returned tensor has shape (B,) for batches or a single value for single images.
    
    Note:
    This function uses no gradients (`torch.no_grad()`) for its computations, making it suitable for
    use in evaluation or analysis without affecting the computational graph.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
   
    img_tensor = img_tensor.view(img_tensor.size(0), img_tensor.size(1), -1)
    u = img_tensor.mean(dim=1)
    
    # Calculate NMI as the ratio of the 50th percentile (median) intensity to the 95th percentile intensity
    nmi = torch.quantile(u, 0.5, dim=-1) / torch.quantile(u, 0.95, dim=-1)
    
    return nmi

    




