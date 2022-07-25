import numpy as np
import torch
p =4



class SVR_mine(torch.nn.Module):
    def __init__(self,x, y):
        super(SVR_mine, self).__init__()
        self.x = x

        self.x_torch = torch.as_tensor(self.x).float()
        self.y = y

        self.n = x.shape[0]
        K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i, j] = np.power(1 + x[i, :].T @ x[j, :], p)
        self.K = K

        self.alpha_torch = torch.nn.Parameter(torch.from_numpy(np.random.uniform(0.0,0.2,(self.n))),requires_grad=True)
        param_name = 'alpha_torch'
        self.register_parameter(param_name, self.alpha_torch)

        self.b_torch = torch.nn.Parameter(torch.from_numpy(np.random.uniform(0.0,0.2,(1))),requires_grad=True)
        param_name = 'b_torch'
        self.register_parameter(param_name, self.b_torch)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)


    def pred_np(self, x_p):

        f_pred = 0
        for i in range(self.n):
            f_pred = f_pred + self.alpha[i] * rbf(x_p, self.x[i, :])
        f_pred = f_pred + self.b

        return f_pred

    def pred_torch(self,x_p):

        f_pred = 0
        for i in range(self.n):
            f_pred = f_pred + self.alpha_torch[i] * rbf_torch(x_p, self.x_torch[i, :])
        f_pred = f_pred + self.b_torch

        return f_pred.float()


    def loadModel(self,file):
        self.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
        self.eval()

    def saveModel(self,PATH):

        torch.save(self.state_dict(), PATH)


def rbf(x,x_i):
    #k = np.exp((-1/(2*np.power(sig,2))) * np.power(np.linalg.norm(x - x_i), 2))

    k = np.power(1 + x.T @ x_i, p)
    #k = (1 + np.sqrt(3)* (np.linalg.norm(x - x_i)/sig))* np.exp(-np.sqrt(3)*(np.linalg.norm(x - x_i)/sig))
    return k


def svr(alpha,b, x,x_i):

    n = x_i.shape[0]
    f_pred = 0
    for i in range(n):
        f_pred = f_pred + alpha[i]*rbf(x, x_i[i,:])
    f_pred = f_pred+ b

    return f_pred


def rbf_torch(x,x_i):
    #k = torch.exp((-1/(2*np.power(sig,2))) * torch.pow(torch.linalg.norm(x - x_i), 2))

    k = torch.pow(1 + x.T @ x_i, p)
    #k = (1 + np.sqrt(3)* (np.linalg.norm(x - x_i)/sig))* np.exp(-np.sqrt(3)*(np.linalg.norm(x - x_i)/sig))
    return k

def svr_torch(alpha,b, x,x_i):

    n = x_i.shape[0]
    f_pred = 0
    for i in range(n):
        f_pred = f_pred + alpha[i]*rbf_torch(x, x_i[i,:])
    f_pred = f_pred+ b

    return f_pred

