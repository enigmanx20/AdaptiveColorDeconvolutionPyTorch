# implementation based on https://www.sciencedirect.com/science/article/pii/S0169260718312161?via%3Dihub
import torch
import torch.nn.functional as F
import torch.optim as optim
from cd_torch import rgb_from_hed, hed_from_rgb, separate_stains, combine_stains

rgb_from_hed = rgb_from_hed

def calculate_sca_matrix(alphas, betas):
    M = torch.stack([ torch.cos(alphas) * torch.sin(betas), 
                torch.cos(alphas) * torch.cos(betas),
                torch.sin(alphas)]).T
    return M

def loss_fn(hed, lambda_p=0.002, lambda_b=10.0, lambda_e=1.0, gamma=0.3, eta=0.6, eps=1e-5):
    """
    bchw
    """
    h = hed[:, 0]
    e = hed[:, 1]
    d = hed[:, 2]
    
    lp = d**2 + lambda_p * 2 * (h*e)/(h**2 + e**2 + eps)
    lp = lp.mean()
    
    lb = (1-eta) * h.mean() - eta * e.mean()
    lb = lb**2
    
    le = gamma - (h+e).mean()
    le = le**2
    
    loss = lp + lambda_b * lb + lambda_e * le
    return loss

def acd(img_tensor, device, lr=0.001, itr=300, rgb_from_hed=rgb_from_hed):
    rgb_from_hed = rgb_from_hed.to(device)
    _W = torch.nn.Parameter(torch.eye(3)[:2])
    alphas = torch.nn.Parameter(torch.asin(rgb_from_hed[:, 2])).to(device)
    betas = torch.nn.Parameter(torch.asin(rgb_from_hed[:, 0]/torch.cos(alphas))).to(device)
    
    optimizer = optim.Adagrad([_W, alphas, betas], lr=lr)
    optimizer.zero_grad()
    
    for i in range(itr):
        M = calculate_sca_matrix(alphas, betas)
        D = torch.linalg.inv(M)
        W = torch.cat([_W, torch.eye(3)[2].unsqueeze(0)]).to(device)
        hed = separate_stains(img_tensor, torch.matmul(W, D))
        loss = loss_fn(hed).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        M = calculate_sca_matrix(alphas, betas)
        D = torch.linalg.inv(M)
        W = torch.cat([_W, torch.eye(3)[2].unsqueeze(0)]).to(device)
    return M.clone().detach(), D.clone().detach(), W.clone().detach()
    
    