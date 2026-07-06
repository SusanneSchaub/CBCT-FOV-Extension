import torch
import torch.nn.functional as F

def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_mse_loss_no_add( x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss = {"loss": 0.}
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] = loss_mse
    return loss


def calc_l1_loss_no_add( x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss = {"loss_l1": 0.}
    loss_l1 = torch.mean(torch.abs(x-y))
    loss["loss_l1"] = loss_l1
    return loss



def calc_mse_loss_volume(x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss = {"loss": 0.}
    loss_mse_3d = F.mse_loss(x, y)
    loss["loss"] = loss_mse_3d
    return loss


def calc_tv_loss(x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """

    loss = {"loss_tv": 0.}

    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)

    loss["loss_tv"] = tv * k
    return loss


def calc_tv_loss_GPT(x, k):

    nx, ny, nz = x.shape

    dx = x[1:, :, :] - x[:-1, :, :]
    dy = x[:, 1:, :] - x[:, :-1, :]
    dz = x[:, :, 1:] - x[:, :, :-1]

    # slice to overlapping volume
    dx = dx[:, :ny-1, :nz-1]
    dy = dy[:nx-1, :, :nz-1]
    dz = dz[:nx-1, :ny-1, :]

    tv = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8).mean()

    return {"loss_tv": k * tv}


def calc_second_derivative_loss(x, k):



    dxx = x[2:, :, :] - 2*x[1:-1,:,:] + x[:-2,:,:]
    dyy = x[:, 2:, :] - 2*x[:,1:-1,:] + x[:,:-2,:]
    dzz = x[:, :, 2:] - 2*x[:,:,1:-1] + x[:,:,:-2]

    loss = (
        dxx.pow(2).mean() +
        dyy.pow(2).mean() +
        dzz.pow(2).mean()
    )

    return {"loss_second": k * loss}



