"""
Darcy flow problem
8 cases...
primal/mixed + fc/conv + variational/residual
"""
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')


def grad(outputs, inputs):
    return ag.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
                   create_graph=True)


def bilinear_interpolate_torch(im, x, y):
    # https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ][0]
    Ib = im[ y1, x0 ][0]
    Ic = im[ y0, x1 ][0]
    Id = im[ y1, x1 ][0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) \
        + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)


def primal_residual_fc(model, x, K_grad_ver, K_grad_hor, K, verbose=False):
    """Computes the residules of satisifying PDE at x
    Permeability is also provided: K

    First assume x is on grid

    Args:
        model (Module): u = f(x) pressure network, input is spatial coordinate
        x (Tensor): (N, 2) spatial input x. could be off-grid, vary very pass
        grad_K (Tensor): estimated gradient field
        verbose (bool): If True, print info
    Returns:
        residual: (N, 1)
    """
 
    assert len(K_grad_ver) == len(x)
    x.requires_grad = True
    u = model(x)
    # grad outputs a tuple: (N, 2)
    u_x = grad(u, x)[0]
    
    div1 = K_grad_ver * u_x[:, 0] + K * grad(u_x[:, 0], x)[0][:, 0]
    div2 = K_grad_hor * u_x[:, 1] + K * grad(u_x[:, 1], x)[0][:, 1]
    div = div1 + div2

    if verbose:
        print(div.detach().mean(), div.detach().max(), div.detach().min())
    return (div ** 2).mean()

def neumann_boundary(model, x):
    # bug: u_y! NOT u_x
    x.requires_grad = True
    u = model(x)
    u_ver = grad(u, x)[0][:, 0]
    return (u_ver ** 2).mean()


def neumann_boundary_mixed(model, x):

    # x.requires_grad = True
    y = model(x)
    tau_ver = y[:, 1]
    
    return (tau_ver ** 2).mean()


def primal_variational_fc(model, x, K, verbose=False):
    """Evaulate energy functional. Simple MC. Evaluate on [1:-1, 1:-1] of grid

    Args:
        x (Tensor): colloc points on interior of grid (63 ** 2, 2)
    """
    x.requires_grad = True
    u = model(x)
    u_x = grad(u, x)[0]
    u_x_squared = (u_x ** 2).sum(1)
    energy = (0.5 * K * u_x_squared).mean()
    if verbose:
        print(f'energy: {energy:.6f}')
    return energy


def mixed_residual_fc(model, x, K, verbose=False, rand_colloc=False, fig_dir=None):
    """
    Args:
        x: (N, 2)
        K: (N, 1)

    """
    x.requires_grad = True
    # (N, 3)
    y = model(x)
    u = y[:, 0]
    # (N, 2)
    tau = y[:, [1, 2]]
    # (N, 2)
    u_x = grad(u, x)[0]

    grad_tau_ver = grad(y[:, 1], x)[0][:, 0]
    grad_tau_hor = grad(y[:, 2], x)[0][:, 1]

    if rand_colloc:
        K = bilinear_interpolate_torch(K.unsqueeze(-1), x[:, [1]], x[:, [0]])
        K = K.t()
        # print(f'K interp: {K.shape}')
        # plt.imshow(K[0].detach().cpu().numpy().reshape(65, 65))
        # plt.savefig(fig_dir+'/Kinterp.png')
        # plt.close()


    loss_constitutive = ((K * u_x + tau) ** 2).mean()
    loss_continuity = ((grad_tau_ver + grad_tau_hor) ** 2).mean()

    return loss_constitutive + loss_continuity


"""
ConvNet ============================================
"""

def energy_functional_exp(input, output, sobel_filter):
    r""" sigma = -exp(K * u) * grad(u)

    V(u, K) = \int 0.5 * exp(K*u) * |grad(u)|^2 dx
    """
    grad_h = sobel_filter.grad_h(output)
    grad_v = sobel_filter.grad_v(output)

    return (0.5 * torch.exp(input * output) * (grad_h ** 2 + grad_v ** 2)).mean()


def cc2_fe(input, output,device):
    E = 1.0
    nu = 0.3
    # C0np = np.array()
    C0 = E/(1-nu**2)*torch.Tensor([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]).to(device)
    pp = input.contiguous().view(input.shape[0], -1, 1, 1).to(device)
    # pp = input.permute(0,1,3,2).contiguous().view(input.shape[0], -1, 1, 1).to(device)
    C = pp**3*C0
    
    
    ux = output[:,[0]]
    uy = output[:,[1]]
    unfold = nn.Unfold(kernel_size=(2, 2))
    # [bs, 4, w*h]
    uex = unfold(ux)
    uey = unfold(uy)
    ue_not = torch.cat([uex, uey], 1).permute([0,2,1])
    ue = ue_not[:,:,[0,4,1,5,3,7,2,6]].unsqueeze(3)
    
    Bxy=np.zeros([2,3,8],dtype=np.float32)
    Bxy[0,:,:]=np.array([ [0, 0,  0,  0, 0, 0,  0,  0],
                   [0, 1,  0, -1, 0, 1,  0, -1],
                   [1, 0, -1,  0, 1, 0, -1,  0] ])
    Bxy[1,:,:]=np.array([ [1, 0, -1,  0, 1, 0, -1,  0],
                   [0, 0,  0,  0, 0, 0,  0,  0],
                   [0, 1,  0, -1, 0, 1,  0, -1] ])
    Bxy = torch.from_numpy(Bxy).to(device)
    S_x = torch.matmul(C, torch.matmul(Bxy[0,:,:],ue))
    S_y = torch.matmul(C, torch.matmul(Bxy[1,:,:],ue))
    return S_x.shape

def cc1_new(input, output, device):
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    # KE 8x8
    KE = np.array(KE, dtype=np.float32)
    Ke = torch.from_numpy(KE).to(device)
    pp = input.contiguous().view(input.shape[0], -1, 1, 1).to(device)
    # K [bs, 800, 8, 8]
    K = pp**3*Ke
    # ???
    # output[:,:,:,0] = 0
    ux = output[:,[0]]
    uy = output[:,[1]]

    unfold = nn.Unfold(kernel_size=(2, 2))
    # [bs, 4, w*h]
    uex = unfold(ux)
    uey = unfold(uy)
    ue_not = torch.cat([uex, uey], 1).permute([0,2,1])
    ue = ue_not[:,:,[0,4,1,5,3,7,2,6]].unsqueeze(3)

    # KU
    KU = torch.matmul(K, ue)
    tku = KU.permute([0,2,1,3]).contiguous().view(-1,8,20,40)

    result = torch.zeros([ue.shape[0],2,21,41]).to(device)

    result[:,:,:20,:40] = tku[:,[0,1],:,:]
    result[:,:,:20,1:] += tku[:,[2,3],:,:]
    result[:,:,1:,:40] += tku[:,[6,7],:,:]
    result[:,:,1:,1:] += tku[:,[4,5],:,:]

    F = torch.zeros([ue.shape[0],2,21,41]).to(device)
    F[:,0,:,-1] = 1 
    F[:,0,0,-1] = 0.5
    F[:,0,-1,-1] = 0.5
    return ((result[:,:,:,1:]-F[:,:,:,1:])**2).sum([1,2,3]).mean()

def bc_new(output):
    ux = output[:, [0]]
    uy = output[:, [1]]
    lu = ux[:,:,:,0]**2 + uy[:,:,:,0]**2
    return lu.sum([-1]).mean()


def cc1(input, output, output_post, sobel_filter, device):
    E = 1.0
    nu = 0.3
    # C0np = np.array()
    C0 = E/(1-nu**2)*torch.Tensor([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]).to(device)
    pp = input.contiguous().view(input.shape[0], -1, 1, 1).to(device)
    # pp = input.permute(0,1,3,2).contiguous().view(input.shape[0], -1, 1, 1).to(device)
    C = pp**3*C0
    
    

    # duxdx = sobel_filter.grad_h(output[:, [0]])
    # duxdy = sobel_filter.grad_v(output[:, [0]])
    # duydx = sobel_filter.grad_h(output[:, [1]])
    # duydy = sobel_filter.grad_v(output[:, [1]])
    # d1 = duxdx
    # d2 = duydy
    # d3 = duxdy+duydx
    # du = torch.cat([d1,d2,d3],1)
    # du_post = du.view(du.shape[0],3,-1,1).permute(0,2,1,3)
    B=np.zeros([4,3,8],dtype=np.float32)
    B[0,:,:]=np.array([ [-1,  0, 1, 0, 0, 0, 0, 0],
                        [0, -1, 0, 0, 0, 0, 0, 1],
                        [-1, -1, 0, 1, 0, 0, 1, 0] ])
    B[1,:,:]=np.array([ [-1,  0,  1,  0, 0, 0, 0, 0],
                        [0,  0,  0, -1, 0, 1, 0, 0],
                        [0, -1, -1,  1, 1, 0, 0, 0] ])
    B[2,:,:]=np.array([ [0, 0,  0,  0, 1, 0, -1,  0],
                        [0, 0,  0, -1, 0, 1,  0,  0],
                        [0, 0, -1,  0, 1, 1,  0, -1] ])
    B[3,:,:]=np.array([ [0,  0, 0, 0, 1, 0, -1,  0],
                        [0, -1, 0, 0, 0, 0,  0,  1],
                        [-1,  0, 0, 0, 0, 1,  1, -1] ])
    B = torch.from_numpy(B).to(device)

    B0 = torch.Tensor([[-0.5,0,-0.5],[0,-0.5,-0.5],[0.5,0,-0.5],\
        [0,-0.5,0.5],[0.5,0,0.5] ,[0,0.5,0.5],[-0.5,0,0.5],[0,-0.5,-0.5]]).to(device)
    B0T = B0.transpose(0,1)
    ux = output[:,[0]]
    uy = output[:,[1]]
    
    k1 = torch.FloatTensor( np.array([[[[-0.5,0.5],[-0.5,0.5]]]]) ).to(device)
    k2 = torch.FloatTensor( np.array([[[[-0.5,-0.5],[-0.5,0.5]]]]) ).to(device)
    k31 = torch.FloatTensor( np.array([[[[-0.5,-0.5],[0.5,0.5]]]]) ).to(device)
    k32 = torch.FloatTensor( np.array([[[[-0.5,0.5],[-0.5,0.5]]]]) ).to(device)
    ux1 = F.conv2d(ux, k1, stride=1, padding=0, bias=None)
    uy1 = F.conv2d(uy, k2, stride=1, padding=0, bias=None)
    ux2 = F.conv2d(ux, k31, stride=1, padding=0, bias=None)
    uy2 = F.conv2d(uy, k32, stride=1, padding=0, bias=None)
    utest = torch.cat([ux1, uy1, ux2+uy2], 1)

    unfold = nn.Unfold(kernel_size=(2, 2))
    # [bs, 4, w*h]
    uex = unfold(ux)
    uey = unfold(uy)
    ue_not = torch.cat([uex, uey], 1).permute([0,2,1])
    ue = ue_not[:,:,[0,4,1,5,3,7,2,6]].unsqueeze(3)


    # sig = output_post[:, [2,3,4]]
    # sig_post = sig.view(sig.shape[0],3,-1,1).permute(0,2,1,3)
    du0 = torch.matmul(C, torch.matmul(B[0,:,:],ue))
    du1 = torch.matmul(C, torch.matmul(B[1,:,:],ue))
    du2 = torch.matmul(C, torch.matmul(B[2,:,:],ue))
    du3 = torch.matmul(C, torch.matmul(B[3,:,:],ue))
    du0t = du0.permute([0,2,1,3]).contiguous().view(-1,3,20,40)
    du1t = du1.permute([0,2,1,3]).contiguous().view(-1,3,20,40)
    du2t = du2.permute([0,2,1,3]).contiguous().view(-1,3,20,40)
    du3t = du3.permute([0,2,1,3]).contiguous().view(-1,3,20,40)
    ones = torch.ones_like(du0t)

    masks =torch.zeros([du0.shape[0],3,21,41]).to(device)
    result =torch.zeros([du0.shape[0],3,21,41]).to(device)
    result[:,:,:20,:40] = du0t
    result[:,:,:20,1:] += du1t
    result[:,:,1:,:40] += du3t
    result[:,:,1:,1:] += du2t

    masks[:,:,:20,:40] += ones
    masks[:,:,:20,1:] += ones
    masks[:,:,1:,:40] += ones
    masks[:,:,1:,1:] += ones
    lp1 = (result / masks) - output[:, [2,3,4]]
    # lp1 = torch.matmul(C,du_post) - sig_post
    # lp1 = torch.matmul(C, torch.matmul(B0T,ue)) - sig_post
    # ???
    # return (lp1**2).sum([1,2,3])
    return (lp1**2).sum([1,2,3]).mean()


# !!!!!!!!!!!!
def conv_constitutive_constraint(input, output, sobel_filter):
    """sigma = - K * grad(u)

    Args:
        input (Tensor): (1, 1, 65, 65)
        output (Tensor): (1, 3, 65, 65), 
            three channels from 0-2: u, sigma_1, sigma_2
    """
    grad_h = sobel_filter.grad_h(output[:, [0]])
    grad_v = sobel_filter.grad_v(output[:, [0]])
    est_sigma1 = - input * grad_h
    est_sigma2 = - input * grad_v

    return ((output[:, [1]] - est_sigma1) ** 2 
        + (output[:, [2]] - est_sigma2) ** 2).mean()


def conv_constitutive_constraint_nonlinear(input, output, sobel_filter, beta1, beta2):
    """Nonlinear extension of Darcy's law
        - K * grad_u = sigma + beta1 * sqrt(K) * sigma ** 2 + beta2 * K * sigma ** 3

    Args:
        input: K
        output: u, sigma1, sigma2
    """
    K_u_h = - input * sobel_filter.grad_h(output[:, [0]])
    K_u_v = - input * sobel_filter.grad_v(output[:, [0]])
    sigma = output[:, [1, 2]]
    rhs = sigma + beta1 * torch.sqrt(input) * sigma ** 2 + beta2 * input * sigma ** 3
    return ((K_u_h - rhs[:, [0]])** 2 + (K_u_v - rhs[:, [1]]) ** 2).mean()

def conv_constitutive_constraint_nonlinear_exp(input, output, sobel_filter):
    """Nonlinear extension of Darcy's law
        sigma = - exp(K * u) grad(u)

    Args:
        input: K
        output: u, sigma1, sigma2
    """
    grad_h = sobel_filter.grad_h(output[:, [0]])
    grad_v = sobel_filter.grad_v(output[:, [0]])

    sigma_h = - torch.exp(input * output[:, [0]]) * grad_h
    sigma_v = - torch.exp(input * output[:, [0]]) * grad_v

    return ((output[:, [1]] - sigma_h) ** 2 
        + (output[:, [2]] - sigma_v) ** 2).mean()


def cc2new(output, sobel_filter):
    sx = output[:, [2]]
    sy = output[:, [3]]
    sxy = output[:, [4]]
    dsxdx = sx[:,:,:20,1:]-sx[:,:,:20,:40]
    # dsxdy = sx[:,:,1:,:40]-sx[:,:,:20,:40]
    dsydy = sy[:,:,1:,:40]-sy[:,:,:20,:40]
    dsxydx = sxy[:,:,:20,1:]-sxy[:,:,:20,:40]
    dsxydy = sxy[:,:,1:,:40]-sxy[:,:,:20,:40]
    ds = torch.cat([dsxdx+dsxydy, dsydy+dsxydx],1)
    return (ds ** 2).sum([1,2,3]).mean()


# !!!!!!!!!!!!
def cc2(output, sobel_filter):
    dsxdx = sobel_filter.grad_h(output[:, [2]])
    dsydy = sobel_filter.grad_v(output[:, [3]])
    dsxydx = sobel_filter.grad_h(output[:, [4]])
    dsxydy = sobel_filter.grad_v(output[:, [4]])

    ds = torch.cat([dsxdx+dsxydy, dsydy+dsxydx],1)
#     return (ds ** 2).mean()
    #return (ds ** 2).sum([1,2,3]).mean()
    return (ds ** 2).sum([2,3]).mean()


# !!!!!!!!!!!!
def conv_continuity_constraint(output, sobel_filter, use_tb=True):
    """
    div(sigma) = -f

    Args:

    """
    sigma1_x1 = sobel_filter.grad_h(output[:, [1]])
    sigma2_x2 = sobel_filter.grad_v(output[:, [2]])
    # leave the top and bottom row free, since sigma2_x2 is almost 0,
    # don't want to enforce sigma1_x1 to be also zero.
    if use_tb:
        return ((sigma1_x1 + sigma2_x2) ** 2).mean()
    else:
        return ((sigma1_x1 + sigma2_x2) ** 2)[:, :, 1:-1, :].mean()

def bc(output, output_post):
    ux = output[:, [0]]
    uy = output[:, [1]]
    lu = ux[:,:,:,0]**2 + uy[:,:,:,0]**2
    sx = output[:, [2]]
    sy = output[:, [3]]
    sxy = output[:, [4]]
    lbr = (sx[:,:,:,40]-1)**2 + (sxy[:,:,:,40])**2
    lbt = (sy[:,:,0,2:-2])**2 + (sxy[:,:,0,2:-2])**2
    lbb = (sy[:,:,20,2:-2])**2 + (sxy[:,:,20,2:-2])**2
    return torch.cat([lu,lbb,lbt,lbr],2).sum([-1]).mean()

# !!!!!!!!!!!!
def conv_boundary_condition(output):
    left_bound, right_bound = output[:, 0, :, 0], output[:, 0, :, -1]
    top_down_flux = output[:, 2, [0, -1], :]
    loss_dirichlet = F.mse_loss(left_bound, torch.ones_like(left_bound)) \
        + F.mse_loss(right_bound, torch.zeros_like(right_bound))
    loss_neumann = F.mse_loss(top_down_flux, torch.zeros_like(top_down_flux))

    return loss_dirichlet, loss_neumann
