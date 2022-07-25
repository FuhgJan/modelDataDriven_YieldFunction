import numpy as np
import torch


def reshape_fortran(x, shape):
    # Fortran like vector reshape
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

class GP_with_torch(torch.nn.Module):
    def __init__(self,_Ssc,S,theta,beta,gamma,Ysc,m,n):
        super(GP_with_torch, self).__init__()
        self.Ssc = torch.as_tensor(_Ssc).double()
        self.S = torch.as_tensor(S).double()
        self.theta =  torch.as_tensor(theta).double()
        self.beta = torch.as_tensor(beta).double()
        self.gamma = torch.as_tensor(gamma).double()
        self.Ysc = torch.as_tensor(Ysc).double()
        self.m = m
        self.n = n

    def predict_torch(self, inp_data):
        x = inp_data
        x = (x - self.Ssc[[0], :]) / self.Ssc[[1], :]
        # Get distance to design sites
        m, n = self.m, self.n
        mx, nx = x.shape
        dx = torch.zeros((mx * m, n)).double()
        kk = torch.arange(m).reshape(1, -1)

        for k in torch.arange(mx):
            dx[kk, :] = x[k, :] - self.S
            kk = kk + m

        # Get regression function or correlation
        mm, nn = x.shape
        f = torch.ones((mm, 1)).double()

        m_d, n_d = dx.shape  # number of differences and dimension of data

        if self.theta.shape[0] == 1:
            self.theta = torch.tile(self.theta, (1, n_d))
        elif self.theta.shape[0]!= n:
            raise ValueError(f'Length of theta must be 1 or {n_d}.')

        kval = (np.sqrt(3) * torch.abs(dx)) *self.theta[:, 0].repeat(m_d,1)
        val = (1 + kval) * torch.exp(-kval)
        r2 = torch.prod(val, axis=1)

        r = reshape_fortran(r2, (m, mx))
        # scaled predictor
        sy = f @ self.beta + (self.gamma @ r).T

        # predictor
        y = self.Ysc[[0], :] + self.Ysc[[1], :] * sy

        return y





