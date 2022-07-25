"""
@author: Jan N. Fuhg, Cornell University, 2022

Selected codes for model-data-driven yield functions, including plots for uniaxial, biaxial and 3D yield surface shapes
as well as material response of models with return-mapping algorithm
"""


import scipy.io as sio
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import mDataDriven.ICNN as icnn
import mDataDriven.normalNN as Nnn
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy import spatial
from GP_torch import GP_with_torch
from SVR_1 import SVR_mine
import pickle
from scipy import optimize
from random import uniform
import math
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import pylab
from mpl_toolkits.mplot3d import axes3d, art3d
from torch.autograd.functional import jacobian
import warnings
warnings.filterwarnings("ignore")
# ----------------------------------------------------------------------------------
## Establish global variables
# ----------------------------------------------------------------------------------
sig = 430.384
sigv = sig
F=0.433973
G=0.552709
H=0.447291
N=0.472479
L= 0.
M = 0.
sigz= 0.
sigxz= 0.
sigyz= 0.
a1 = 0.869
a2 = 3.371
a3 = 3.509
a4 = 1
a5 = 0.
a6 = 0.
b1 = -1.591
b2 = 5.414
b3 = 3.957
b4 = 0.259
b5= 1
b10 = 1
b6=0.
b7=0.
b8=0.
b9=0.
cv = 2.01
tau= 300
E = 10000
nu = 0.3
S = (1/E)*torch.tensor(((1,-nu,0),(-nu,1,0),(0,0,1+nu)))
inv_S = torch.inverse(S)


def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0]) * scale
    return mplotlims[1] - offset, mplotlims[0] + offset

def clockwiseangle_and_distance(point):
    origin = [0, 0]
    refvec = [0, 1]

    vector = [point[0]-origin[0], point[1]-origin[1]]
    lenvector = math.hypot(vector[0], vector[1])

    if lenvector == 0:
        return -math.pi, 0
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)

    if angle < 0:
        return 2*math.pi+angle, lenvector
    return angle, lenvector

def fVM_fun_sympy(sigtheta, beta):
    sigx = sigtheta
    sigy = sigtheta*(np.tan(beta*np.pi/ 180))
    fVM = F * (sigy - sigz) ** 2 + G * (sigz - sigx) ** 2 + H * (sigx - sigy) ** 2 + 2 * L * (sigyz) ** 2 + 2 * M * (
        sigxz) ** 2 + 2 * N * (0.) ** 2 - sig ** 2
    return fVM


def fCB_fun_sympy(sigtheta,beta):
    sigx = sigtheta
    sigy = sigtheta*(np.tan(beta*np.pi/ 180))
    J2 = (a1 / 6) * (sigx - sigy) ** 2 + (a2 / 6) * (sigy - sigz) ** 2 + (a3 / 6) * (sigx - sigz) ** 2 + a4 * (
        0.) ** 2 + a5 * (sigxz) ** 2 + a6 * (sigyz) ** 2
    J3 = ((b1 + b2) / 27) * (sigx) ** 3 + ((b3 + b4) / 27) * sigy ** 3 + (1 / 27) * (2 * (b1 + b4) - b2 - b3) * (
        sigz) ** 3 - (1 / 9) * (b1 * sigy + b2 * sigz) * (sigx) ** 2 - (1 / 9) * (b3 * sigz + b4 * sigx) * (
                     sigy ** 2) - (1 / 9) * ((b1 - b2 + b4) * sigx + (b1 - b3 + b4) * sigy) * (sigz ** 2) + (2 / 9) * (
                     b1 + b4) * sigx * sigy * sigz - ((sigxz ** 2) / 3) * (
                     2 * b9 * sigy - b8 * sigz - (2 * b9 - b8) * sigx) - ((0. ** 2) / 3) * (
                     2 * b10 * sigz - b5 * sigy - (2 * b10 - b5) * sigx) - ((sigyz ** 2) / 3) * (
                     (b6 + b7) * sigx - b6 * sigy - b7 * sigz)
    fCB = J2 ** (3 / 2) - cv * J3 - tau ** 3


    return fCB


def correctedModel(sigtheta,beta,net):
    sigx = sigtheta
    sigy = sigtheta*(np.tan(beta*np.pi/ 180))
    fVM = fVM_fun_sympy(sigtheta,beta)

    inp = np.array((float(sigx),float(sigy),float(0.0))).reshape(1,3)
    inp_Test_norm = torch.as_tensor(x_scaler_added.transform(inp)).float()

    pred_norm = net(inp_Test_norm).detach().cpu().numpy()
    pred = y_scaler_added.inverse_transform(pred_norm)
    fVM_corrected = fVM+ pred[0][0]
    return fVM_corrected


def correctedModelSVR(sigtheta,beta):
    sigx = sigtheta
    sigy = sigtheta*(np.tan(beta*np.pi/ 180))
    fVM = fVM_fun_sympy(sigtheta,beta)

    inp = np.array((float(sigx),float(sigy),float(0.0))).reshape(1,3)
    inp_Test_norm = x_scaler_added.transform(inp)

    pred_norm = svr_corrected.pred_np(inp_Test_norm.T).reshape(-1,1)
    pred = y_scaler_added.inverse_transform(pred_norm)
    fVM_corrected = fVM+ pred[0][0]
    return fVM_corrected


def correctedModelGP(sigtheta,beta):
    sigx = sigtheta
    sigy = sigtheta*(np.tan(beta*np.pi/ 180))
    fVM = fVM_fun_sympy(sigtheta,beta)

    inp = np.array((float(sigx),float(sigy),float(0.0))).reshape(1,3)
    pred = gp_corrected.predict_torch(torch.as_tensor(inp).float()).detach().cpu().numpy()

    fVM_corrected = fVM+ pred[0][0]
    return fVM_corrected



def netModel(sigtheta,beta,net):
    sigx = sigtheta
    sigy = sigtheta*(np.tan(beta*np.pi/ 180))

    inp = np.array((float(sigx),float(sigy),float(0.0))).reshape(1,3)
    inp_Test_norm = torch.as_tensor(x_scaler.transform(inp)).float()

    pred = net(inp_Test_norm).detach().cpu().numpy()
    fVM_corrected = pred[0][0]
    return fVM_corrected



def uni_fVM_fun(sigtheta, beta):
    sigx = sigtheta*(np.cos(beta*np.pi/ 180)**2)
    sigy = sigtheta*(np.sin(beta*np.pi/ 180)**2)
    sigxy = sigtheta*np.cos(beta*np.pi/ 180)*np.sin(beta*np.pi/ 180)
    fVM = F * (sigy - sigz) ** 2 + G * (sigz - sigx) ** 2 + H * (sigx - sigy) ** 2 + 2 * L * (sigyz) ** 2 + 2 * M * (
        sigxz) ** 2 + 2 * N * (sigxy) ** 2 - sig ** 2
    return fVM

def uni_fCB_fun(sigtheta,beta):
    sigx = sigtheta*(np.cos(beta*np.pi/ 180)**2)
    sigy = sigtheta*(np.sin(beta*np.pi/ 180)**2)
    sigxy = sigtheta*np.cos(beta*np.pi/ 180)*np.sin(beta*np.pi/ 180)
    J2 = (a1 / 6) * (sigx - sigy) ** 2 + (a2 / 6) * (sigy - sigz) ** 2 + (a3 / 6) * (sigx - sigz) ** 2 + a4 * (
        sigxy) ** 2 + a5 * (sigxz) ** 2 + a6 * (sigyz) ** 2
    J3 = ((b1 + b2) / 27) * (sigx) ** 3 + ((b3 + b4) / 27) * sigy ** 3 + (1 / 27) * (2 * (b1 + b4) - b2 - b3) * (
        sigz) ** 3 - (1 / 9) * (b1 * sigy + b2 * sigz) * (sigx) ** 2 - (1 / 9) * (b3 * sigz + b4 * sigx) * (
                     sigy ** 2) - (1 / 9) * ((b1 - b2 + b4) * sigx + (b1 - b3 + b4) * sigy) * (sigz ** 2) + (2 / 9) * (
                     b1 + b4) * sigx * sigy * sigz - ((sigxz ** 2) / 3) * (
                     2 * b9 * sigy - b8 * sigz - (2 * b9 - b8) * sigx) - ((sigxy ** 2) / 3) * (
                     2 * b10 * sigz - b5 * sigy - (2 * b10 - b5) * sigx) - ((sigyz ** 2) / 3) * (
                     (b6 + b7) * sigx - b6 * sigy - b7 * sigz)
    fCB = J2 ** (3 / 2) - cv * J3 - tau ** 3


    return fCB

def uni_correctedModelGP(sigtheta,beta):
    sigx = sigtheta*(np.cos(beta*np.pi/ 180)**2)
    sigy = sigtheta*(np.sin(beta*np.pi/ 180)**2)
    sigxy = sigtheta*np.cos(beta*np.pi/ 180)*np.sin(beta*np.pi/ 180)
    fVM = uni_fVM_fun(sigtheta,beta)

    inp = np.array((float(sigx),float(sigy),float(sigxy))).reshape(1,3)

    pred = gp_corrected.predict_torch(torch.as_tensor(inp).float()).detach().cpu().numpy()
    fVM_corrected = fVM+ pred[0][0]
    return fVM_corrected



def uni_correctedModelSVR(sigtheta,beta):
    sigx = sigtheta*(np.cos(beta*np.pi/ 180)**2)
    sigy = sigtheta*(np.sin(beta*np.pi/ 180)**2)
    sigxy = sigtheta*np.cos(beta*np.pi/ 180)*np.sin(beta*np.pi/ 180)
    fVM = uni_fVM_fun(sigtheta,beta)

    inp = np.array((float(sigx),float(sigy),float(sigxy))).reshape(1,3)
    inp_Test_norm = x_scaler_added.transform(inp)

    pred_norm = svr_corrected.pred_np(inp_Test_norm.T).reshape(-1,1)
    pred = y_scaler_added.inverse_transform(pred_norm)
    fVM_corrected = fVM+ pred[0][0]
    return fVM_corrected



def uni_correctedModel(sigtheta,beta,net):
    sigx = sigtheta*(np.cos(beta*np.pi/ 180)**2)
    sigy = sigtheta*(np.sin(beta*np.pi/ 180)**2)
    sigxy = sigtheta*np.cos(beta*np.pi/ 180)*np.sin(beta*np.pi/ 180)
    fVM = uni_fVM_fun(sigtheta,beta)

    inp = np.array((float(sigx),float(sigy),float(sigxy))).reshape(1,3)
    inp_Test_norm = torch.as_tensor(x_scaler_added.transform(inp)).float()

    pred_norm = net(inp_Test_norm).detach().cpu().numpy()
    pred = y_scaler_added.inverse_transform(pred_norm)
    fVM_corrected = fVM+ pred[0][0]
    return fVM_corrected



def fVM_fun(x):
    sigx = x[:,0]
    sigy = x[:,1]
    sigxy = x[:,2]
    fVM = F * (sigy - sigz) ** 2 + G * (sigz - sigx) ** 2 + H * (sigx - sigy) ** 2 + 2 * L * (sigyz) ** 2 + 2 * M * (
        sigxz) ** 2 + 2 * N * (sigxy) ** 2 - sig ** 2
    return fVM

def fVMCorrected_fun(x):


    fVM = fVM_fun(x)


    inp_Test_norm = torch.as_tensor(x_scaler_added.transform(x)).float()

    pred_norm = icnn_corrected(inp_Test_norm).detach().cpu().numpy()
    pred = y_scaler_added.inverse_transform(pred_norm)
    fVM_corrected = fVM+ pred[:,0]

    return fVM_corrected



def net_fun_NN(x):


    inp_Test_norm = torch.as_tensor(x_scaler.transform(x)).float()

    pred_norm = net_pureNN(inp_Test_norm).detach().cpu().numpy()

    fVM_corrected =  pred_norm[:,0]

    return fVM_corrected

def net_fun(x):


    inp_Test_norm = torch.as_tensor(x_scaler.transform(x)).float()

    pred_norm = net_pure(inp_Test_norm).detach().cpu().numpy()

    fVM_corrected =  pred_norm[:,0]

    return fVM_corrected

def fVMCorrected_SVRfun(x):

    fVM = fVM_fun(x)


    inp_Test_norm = x_scaler_added.transform(x)

    pred_norm = svr_corrected.pred_np(inp_Test_norm.T).reshape(-1,1)
    pred = y_scaler_added.inverse_transform(pred_norm)
    fVM_corrected = fVM+ pred[:,0]
    return fVM_corrected


def fVMCorrected_GPfun(x):

    fVM = fVM_fun(x)

    pred = gp_corrected.predict_torch(torch.as_tensor(x).double()).detach().cpu().numpy()
    fVM_corrected = fVM+ pred[:,0]

    return fVM_corrected


def fCB_fun(x):
    sigx = x[:,0]
    sigy = x[:,1]
    sigxy = x[:,2]
    J2 = (a1 / 6) * (sigx - sigy) ** 2 + (a2 / 6) * (sigy - sigz) ** 2 + (a3 / 6) * (sigx - sigz) ** 2 + a4 * (
        sigxy) ** 2 + a5 * (sigxz) ** 2 + a6 * (sigyz) ** 2
    J3 = ((b1 + b2) / 27) * (sigx) ** 3 + ((b3 + b4) / 27) * sigy ** 3 + (1 / 27) * (2 * (b1 + b4) - b2 - b3) * (
        sigz) ** 3 - (1 / 9) * (b1 * sigy + b2 * sigz) * (sigx) ** 2 - (1 / 9) * (b3 * sigz + b4 * sigx) * (
                     sigy ** 2) - (1 / 9) * ((b1 - b2 + b4) * sigx + (b1 - b3 + b4) * sigy) * (sigz ** 2) + (2 / 9) * (
                     b1 + b4) * sigx * sigy * sigz - ((sigxz ** 2) / 3) * (
                     2 * b9 * sigy - b8 * sigz - (2 * b9 - b8) * sigx) - ((sigxy ** 2) / 3) * (
                     2 * b10 * sigz - b5 * sigy - (2 * b10 - b5) * sigx) - ((sigyz ** 2) / 3) * (
                     (b6 + b7) * sigx - b6 * sigy - b7 * sigz)
    fCB = J2 ** (3 / 2) - cv * J3 - tau ** 3

    return fCB




def run_uni():
    sigx_val_ten = []
    sigx_val_com = []
    for i in range(betavec0_plot.shape[0]):

        beta = betavec0_plot[i]
        sigtheta_sym = sym.Symbol('sigtheta')

        sigx = sigtheta_sym * (sym.cos(beta * np.pi / 180) ** 2)
        sigy = sigtheta_sym * (sym.sin(beta * np.pi / 180) ** 2)
        sigxy = sigtheta_sym * sym.cos(beta * np.pi / 180) * sym.sin(beta * np.pi / 180)

        J2 = (a1 / 6) * (sigx - sigy) ** 2 + (a2 / 6) * (sigy - sigz) ** 2 + (a3 / 6) * (sigx - sigz) ** 2 + a4 * (
            sigxy) ** 2 + a5 * (sigxz) ** 2 + a6 * (sigyz) ** 2
        J3 = ((b1 + b2) / 27) * (sigx) ** 3 + ((b3 + b4) / 27) * sigy ** 3 + (1 / 27) * (2 * (b1 + b4) - b2 - b3) * (
            sigz) ** 3 - (1 / 9) * (b1 * sigy + b2 * sigz) * (sigx) ** 2 - (1 / 9) * (b3 * sigz + b4 * sigx) * (
                     sigy ** 2) - (1 / 9) * ((b1 - b2 + b4) * sigx + (b1 - b3 + b4) * sigy) * (sigz ** 2) + (2 / 9) * (
                     b1 + b4) * sigx * sigy * sigz - ((sigxz ** 2) / 3) * (
                     2 * b9 * sigy - b8 * sigz - (2 * b9 - b8) * sigx) - ((sigxy ** 2) / 3) * (
                     2 * b10 * sigz - b5 * sigy - (2 * b10 - b5) * sigx) - ((sigyz ** 2) / 3) * (
                     (b6 + b7) * sigx - b6 * sigy - b7 * sigz)
        fCB = J2 ** (3 / 2) - cv * J3 - tau ** 3
        ff = sym.solve(fCB)

        sigx_val_ten.append(ff[1])
        sigx_val_com.append(ff[0])

    uni_sig_vec_fVM_ten = []
    uni_sig_vec_fVM_com = []
    uni_beta_fVM_ten = []
    uni_beta_fVM_com = []


    uni_sig_vec_fVM_corrected_GP_ten = []
    uni_sig_vec_fVM_corrected_GP_com = []
    uni_beta_fVM_corrected_GP_ten = []
    uni_beta_fVM_corrected_GP_com = []

    uni_sig_vec_fVM_corrected_SVR_ten = []
    uni_sig_vec_fVM_corrected_SVR_com = []
    uni_beta_fVM_corrected_SVR_ten = []
    uni_beta_fVM_corrected_SVR_com = []

    uni_sig_vec_fVM_corrected_neti_ten = []
    uni_sig_vec_fVM_corrected_neti_com = []
    uni_beta_fVM_corrected_neti_ten = []
    uni_beta_fVM_corrected_neti_com = []


    uni_sig_vec_fCB_ten = []
    uni_sig_vec_fCB_com = []
    uni_beta_fCB_ten = []
    uni_beta_fCB_com = []


    for i in range(betavec0_true.shape[0]):

        beta = betavec0_true[i]

        x01 = uniform(-500., -100.)
        x02 = uniform(200., 500.)
        r = uniform(0, 1)
        x0 = x01
        if r > 0.5:
            x0 = x02

        root = optimize.newton(uni_correctedModel, x0, args=[beta, icnn_corrected], disp=False)
        sig_t = root
        if root > 0.:
            uni_sig_vec_fVM_corrected_neti_ten.append(sig_t)
            uni_beta_fVM_corrected_neti_ten.append([beta])
        else:
            uni_sig_vec_fVM_corrected_neti_com.append(sig_t)
            uni_beta_fVM_corrected_neti_com.append([beta])

        root = optimize.newton(uni_correctedModelSVR, x0, args=[beta], disp=False)
        sig_t = root
        if root > 0.:
            uni_sig_vec_fVM_corrected_SVR_ten.append(sig_t)
            uni_beta_fVM_corrected_SVR_ten.append([beta])
        else:
            uni_sig_vec_fVM_corrected_SVR_com.append(sig_t)
            uni_beta_fVM_corrected_SVR_com.append([beta])

        root = optimize.newton(uni_correctedModelGP, x0, args=[beta], disp=False)
        sig_t = root
        if root > 0.:
            uni_sig_vec_fVM_corrected_GP_ten.append(sig_t)
            uni_beta_fVM_corrected_GP_ten.append([beta])
        else:
            uni_sig_vec_fVM_corrected_GP_com.append(sig_t)
            uni_beta_fVM_corrected_GP_com.append([beta])

        root = optimize.newton(uni_fVM_fun, x0, args=[beta], disp=False)
        sig_t = root
        if root > 0.:
            uni_beta_fVM_ten.append(beta)
            uni_sig_vec_fVM_ten.append([sig_t])
        else:
            uni_beta_fVM_com.append(beta)
            uni_sig_vec_fVM_com.append([sig_t])

        root = optimize.newton(uni_fCB_fun, x0, args=[beta], disp=False)
        sig_t = root
        if root > 0.:
            uni_beta_fCB_ten.append(beta)
            uni_sig_vec_fCB_ten.append([sig_t])
        else:
            uni_beta_fCB_com.append(beta)
            uni_sig_vec_fCB_com.append([sig_t])

    uni_sig_vec_fVM_corrected_neti_ten_np = np.asarray(uni_sig_vec_fVM_corrected_neti_ten)
    uni_sig_vec_fVM_corrected_neti_com_np = np.asarray(uni_sig_vec_fVM_corrected_neti_com)

    uni_sig_vec_fVM_corrected_SVR_ten_np = np.asarray(uni_sig_vec_fVM_corrected_SVR_ten)
    uni_sig_vec_fVM_corrected_SVR_com_np = np.asarray(uni_sig_vec_fVM_corrected_SVR_com)

    uni_sig_vec_fVM_corrected_GP_ten_np = np.asarray(uni_sig_vec_fVM_corrected_GP_ten)
    uni_sig_vec_fVM_corrected_GP_com_np = np.asarray(uni_sig_vec_fVM_corrected_GP_com)

    uni_sig_vec_fVM_ten_np = np.asarray(uni_sig_vec_fVM_ten)
    uni_sig_vec_fVM_com_np = np.asarray(uni_sig_vec_fVM_com)
    uni_sig_vec_fCB_ten_np = np.asarray(uni_sig_vec_fCB_ten)
    uni_sig_vec_fCB_com_np = np.asarray(uni_sig_vec_fCB_com)

    plt.close()
    fig = plt.figure(figsize=(8, 8))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(uni_beta_fVM_ten, uni_sig_vec_fVM_ten_np, linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(uni_beta_fVM_com, uni_sig_vec_fVM_com_np, linestyle='-', color='k')
    plt.plot(uni_beta_fCB_ten, uni_sig_vec_fCB_ten_np, linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(uni_beta_fCB_com, uni_sig_vec_fCB_com_np, linestyle='--', color='g', lw=2.0)
    plt.plot(uni_beta_fVM_corrected_neti_ten, uni_sig_vec_fVM_corrected_neti_ten_np, linestyle='-', color='b',
             label=r'corrected f', lw=2.0, alpha=0.5)
    plt.plot(uni_beta_fVM_corrected_neti_com, uni_sig_vec_fVM_corrected_neti_com_np, linestyle='-', color='b', lw=2.0,
             alpha=0.5)
    plt.scatter(betavec0_plot, sigx_val_ten, c='k', s=20, zorder=20)
    plt.scatter(betavec0_plot, sigx_val_com, c='k', s=20, zorder=20)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel(r'Angle from rolling direction $\theta[^{\circ}]$', fontsize=18)
    ax.set_ylabel(r'Uniaxial yield stress $\sigma^{u}_{\theta}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5, zorder=1)
    plt.legend(fontsize=13, loc='center left')
    ST = 'mDataDriven/Images/uniData_NetiCorrected_' + str(number) + '.pdf'
    plt.savefig(ST)

    plt.close()
    fig = plt.figure(figsize=(8, 8))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(uni_beta_fVM_ten, uni_sig_vec_fVM_ten_np, linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(uni_beta_fVM_com, uni_sig_vec_fVM_com_np, linestyle='-', color='k')
    plt.plot(uni_beta_fCB_ten, uni_sig_vec_fCB_ten_np, linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(uni_beta_fCB_com, uni_sig_vec_fCB_com_np, linestyle='--', color='g', lw=2.0)
    plt.plot(uni_beta_fVM_corrected_SVR_ten, uni_sig_vec_fVM_corrected_SVR_ten_np, linestyle='-', color='b',
             label=r'corrected f', lw=2.0, alpha=0.5)
    plt.plot(uni_beta_fVM_corrected_SVR_com, uni_sig_vec_fVM_corrected_SVR_com_np, linestyle='-', color='b', lw=2.0,
             alpha=0.5)
    plt.scatter(betavec0_plot, sigx_val_ten, c='k', s=20, zorder=20)
    plt.scatter(betavec0_plot, sigx_val_com, c='k', s=20, zorder=20)
    # ax.axhline(color='k', lw=1.2, ls='-')
    # ax.axvline(color='k', lw=1.2, ls='-')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel(r'Angle from rolling direction $\theta[^{\circ}]$', fontsize=18)
    ax.set_ylabel(r'Uniaxial yield stress $\sigma^{u}_{\theta}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5, zorder=1)
    plt.legend(fontsize=13, loc='center left')
    ST = 'mDataDriven/Images/uniData_SVRCorrected_' + str(number) + '.pdf'
    plt.savefig(ST)

    plt.close()
    fig = plt.figure(figsize=(8, 8))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(uni_beta_fVM_ten, uni_sig_vec_fVM_ten_np, linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(uni_beta_fVM_com, uni_sig_vec_fVM_com_np, linestyle='-', color='k')
    plt.plot(uni_beta_fCB_ten, uni_sig_vec_fCB_ten_np, linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(uni_beta_fCB_com, uni_sig_vec_fCB_com_np, linestyle='--', color='g', lw=2.0)
    plt.plot(uni_beta_fVM_corrected_GP_ten, uni_sig_vec_fVM_corrected_GP_ten_np, linestyle='-', color='b',
             label=r'corrected f', lw=2.0, alpha=0.5)
    plt.plot(uni_beta_fVM_corrected_GP_com, uni_sig_vec_fVM_corrected_GP_com_np, linestyle='-', color='b', lw=2.0,
             alpha=0.5)
    plt.scatter(betavec0_plot, sigx_val_ten, c='k', s=20, zorder=20)
    plt.scatter(betavec0_plot, sigx_val_com, c='k', s=20, zorder=20)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel(r'Angle from rolling direction $\theta[^{\circ}]$', fontsize=18)
    ax.set_ylabel(r'Uniaxial yield stress $\sigma^{u}_{\theta}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5, zorder=1)
    plt.legend(fontsize=13, loc='center left')
    ST = 'mDataDriven/Images/uniData_GPCorrected_' + str(number) + '.pdf'
    plt.savefig(ST)

    plt.close()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(uni_beta_fVM_ten, uni_sig_vec_fVM_ten_np, linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(uni_beta_fVM_com, uni_sig_vec_fVM_com_np, linestyle='-', color='k')
    plt.plot(uni_beta_fCB_ten, uni_sig_vec_fCB_ten_np, linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(uni_beta_fCB_com, uni_sig_vec_fCB_com_np, linestyle='--', color='g', lw=2.0)
    plt.scatter(betavec0_plot, sigx_val_ten, c='k', s=20, zorder=20)
    plt.scatter(betavec0_plot, sigx_val_com, c='k', s=20, zorder=20)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel(r'Angle from rolling direction $\theta[^{\circ}]$', fontsize=18)
    ax.set_ylabel(r'Uniaxial yield stress $\sigma^{u}_{\theta}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5, zorder=1)
    plt.legend(fontsize=13, loc='center left')
    ST = 'mDataDriven/Images/uniData_Pure_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()


def run_biaxial():
    sig_vec_fVM = []
    sig_vec_fVM_corrected_neti = []
    sig_vec_fVM_corrected_GP = []
    sig_vec_fVM_corrected_SVR = []
    sig_vec_fVM_model = []
    sig_vec_fCB = []

    for i in range(betavec0_true.shape[0]):

        beta = betavec0_true[i]

        x01 = uniform(-500., -200.)
        x02 = uniform(200., 500.)
        r = uniform(0, 1)
        x0 = x01
        if r > 0.5:
            x0 = x02

        root = optimize.newton(netModel, x0, args=[beta, net_pure], disp=False)
        sigx = root
        sigy = root * (np.tan(beta * np.pi / 180))
        sig_vec_fVM_model.append([sigx, sigy])

        root = optimize.newton(correctedModelSVR, x0, args=[beta], disp=False)
        sigx = root
        sigy = root * (np.tan(beta * np.pi / 180))
        sig_vec_fVM_corrected_SVR.append([sigx, sigy])

        root = optimize.newton(correctedModelGP, x0, args=[beta], disp=False)
        sigx = root
        sigy = root * (np.tan(beta * np.pi / 180))
        sig_vec_fVM_corrected_GP.append([sigx, sigy])

        root = optimize.newton(correctedModel, x0, args=[beta, icnn_corrected], disp=False)
        sigx = root
        sigy = root * (np.tan(beta * np.pi / 180))
        sig_vec_fVM_corrected_neti.append([sigx, sigy])

        root = optimize.newton(fVM_fun_sympy, x0, args=[beta])
        sigx = root
        sigy = root * (np.tan(beta * np.pi / 180))
        sig_vec_fVM.append([sigx, sigy])

        root = optimize.newton(fCB_fun_sympy, x0, args=[beta], disp=False)
        sigx = root
        sigy = root * (np.tan(beta * np.pi / 180))
        sig_vec_fCB.append([sigx, sigy])

    sig_vec_fVM_corrected_np_neti = np.asarray(sig_vec_fVM_corrected_neti)
    sig_vec_fVM_model_np = np.asarray(sig_vec_fVM_model)
    sig_vec_fVM_corrected_GP_np = np.asarray(sig_vec_fVM_corrected_GP)
    sig_vec_fVM_corrected_SVR_np = np.asarray(sig_vec_fVM_corrected_SVR)
    sig_vec_fVM_np = np.asarray(sig_vec_fVM)
    sig_vec_fCB_np = np.asarray(sig_vec_fCB)

    pts = sig_vec_fVM_corrected_SVR_np.tolist()
    s = sorted(pts, key=clockwiseangle_and_distance)
    sig_vec_fVM_corrected_SVR_np = np.asarray(s)

    sig_vec_fVM_corrected_SVR_np = np.vstack((sig_vec_fVM_corrected_SVR_np, sig_vec_fVM_corrected_SVR_np[0, :]))

    pts = sig_vec_fVM_corrected_GP_np.tolist()
    s = sorted(pts, key=clockwiseangle_and_distance)
    sig_vec_fVM_corrected_GP_np = np.asarray(s)

    sig_vec_fVM_corrected_GP_np = np.vstack((sig_vec_fVM_corrected_GP_np, sig_vec_fVM_corrected_GP_np[0, :]))

    pts = sig_vec_fVM_corrected_np_neti.tolist()
    s = sorted(pts, key=clockwiseangle_and_distance)
    sig_vec_fVM_corrected_np_neti = np.asarray(s)

    sig_vec_fVM_corrected_np_neti = np.vstack((sig_vec_fVM_corrected_np_neti, sig_vec_fVM_corrected_np_neti[0, :]))

    pts = sig_vec_fVM_model_np.tolist()
    s = sorted(pts, key=clockwiseangle_and_distance)
    sig_vec_fVM_model_np = np.asarray(s)

    sig_vec_fVM_model_np = np.vstack((sig_vec_fVM_model_np, sig_vec_fVM_model_np[0, :]))

    pts = sig_vec_fVM_np.tolist()
    s = sorted(pts, key=clockwiseangle_and_distance)
    sig_vec_fVM_np = np.asarray(s)

    sig_vec_fVM_np = np.vstack((sig_vec_fVM_np, sig_vec_fVM_np[0, :]))

    pts = sig_vec_fCB_np.tolist()
    s = sorted(pts, key=clockwiseangle_and_distance)
    sig_vec_fCB_np = np.asarray(s)

    sig_vec_fCB_np = np.vstack((sig_vec_fCB_np, sig_vec_fCB_np[0, :]))

    plt.close()
    fig = plt.figure(figsize=(7, 7))
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(sig_vec_fVM_np[:, 0], sig_vec_fVM_np[:, 1], linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(sig_vec_fCB_np[:, 0], sig_vec_fCB_np[:, 1], linestyle='--', color='g', label=r'true f', lw=2.0)
    ax.scatter(bidata_in[:, 0], bidata_in[:, 1], s=20, c='k')
    ax.axhline(color='k', lw=1.2, ls='-')
    ax.axvline(color='k', lw=1.2, ls='-')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel(r'$\sigma_{x}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{y}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5)
    plt.legend(fontsize=13, loc='upper left')
    ST = 'mDataDriven/Images/biaxialData_Pure_' + str(number) + '.pdf'
    plt.savefig(ST)


    plt.close()
    fig = plt.figure(figsize=(7, 7))
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(sig_vec_fVM_np[:, 0], sig_vec_fVM_np[:, 1], linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(sig_vec_fCB_np[:, 0], sig_vec_fCB_np[:, 1], linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(sig_vec_fVM_corrected_GP_np[:, 0], sig_vec_fVM_corrected_GP_np[:, 1], linestyle='-', color='b',
             label=r'corrected $f$', lw=2.0, alpha=0.5)
    ax.scatter(bidata_in[:, 0], bidata_in[:, 1], s=20, c='k')
    ax.axhline(color='k', lw=1.2, ls='-')
    ax.axvline(color='k', lw=1.2, ls='-')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel(r'$\sigma_{x}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{y}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5)
    plt.legend(fontsize=13, loc='upper left')
    ST = 'mDataDriven/Images/biaxialData_GPCorrected_' + str(number) + '.pdf'
    plt.savefig(ST)

    plt.close()
    fig = plt.figure(figsize=(7, 7))
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(sig_vec_fVM_np[:, 0], sig_vec_fVM_np[:, 1], linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(sig_vec_fCB_np[:, 0], sig_vec_fCB_np[:, 1], linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(sig_vec_fVM_corrected_SVR_np[:, 0], sig_vec_fVM_corrected_SVR_np[:, 1], linestyle='-', color='b',
             label=r'corrected $f$', lw=2.0, alpha=0.5)
    ax.scatter(bidata_in[:, 0], bidata_in[:, 1], s=20, c='k')
    ax.axhline(color='k', lw=1.2, ls='-')
    ax.axvline(color='k', lw=1.2, ls='-')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel(r'$\sigma_{x}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{y}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5)
    plt.legend(fontsize=13, loc='upper left')
    ST = 'mDataDriven/Images/biaxialData_SVRCorrected_' + str(number) + '.pdf'
    plt.savefig(ST)


    plt.close()
    fig = plt.figure(figsize=(7, 7))
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(sig_vec_fVM_np[:, 0], sig_vec_fVM_np[:, 1], linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(sig_vec_fCB_np[:, 0], sig_vec_fCB_np[:, 1], linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(sig_vec_fVM_corrected_np_neti[:, 0], sig_vec_fVM_corrected_np_neti[:, 1], linestyle='-', color='b',
             label=r'corrected $f$', lw=2.0, alpha=0.5)
    ax.scatter(bidata_in[:, 0], bidata_in[:, 1], s=20, c='k')
    ax.axhline(color='k', lw=1.2, ls='-')
    ax.axvline(color='k', lw=1.2, ls='-')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel(r'$\sigma_{x}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{y}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5)
    plt.legend(fontsize=13, loc='upper left')
    ST = 'mDataDriven/Images/biaxialData_ICNNCorrected_' + str(number) + '.pdf'
    plt.savefig(ST)

    plt.close()
    fig = plt.figure(figsize=(7, 7))
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(sig_vec_fVM_np[:, 0], sig_vec_fVM_np[:, 1], linestyle='-', color='k', label=r'model $f_{mod}$')
    plt.plot(sig_vec_fCB_np[:, 0], sig_vec_fCB_np[:, 1], linestyle='--', color='g', label=r'true f', lw=2.0)
    plt.plot(sig_vec_fVM_model_np[:, 0], sig_vec_fVM_model_np[:, 1], linestyle='-', color='b', label=r'fitted $f$',
             lw=2.0, alpha=0.5)
    ax.scatter(bidata_in[:, 0], bidata_in[:, 1], s=20, c='k')
    ax.axhline(color='k', lw=1.2, ls='-')
    ax.axvline(color='k', lw=1.2, ls='-')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel(r'$\sigma_{x}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{y}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5)
    plt.legend(fontsize=13, loc='upper left')
    ST = 'mDataDriven/Images/biaxialData_PureICNN_' + str(number) + '.pdf'
    plt.savefig(ST)
    plt.close()
    # plt.show()



def run_3d():
    n = 75
    a = np.linspace(-2 * sigv, 2 * sigv, n)
    b = np.linspace(-2 * sigv, 2 * sigv, n)
    c = np.linspace(-2 * sigv, 2 * sigv, n)
    X, Y, Z = np.meshgrid(a, b, c)

    XX = X.reshape(-1, 1)
    YY = Y.reshape(-1, 1)
    ZZ = Z.reshape(-1, 1)
    P = np.hstack((XX, YY, ZZ))

    ff_VM = fVM_fun(P).reshape(n, n, n)

    ff_fCB = fCB_fun(P).reshape(n, n, n)
    ff_VMCorrected = fVMCorrected_fun(P).reshape(n, n, n)
    ff_VMCorrectedGP = fVMCorrected_GPfun(P).reshape(n, n, n)
    ff_VMCorrectedSVR = fVMCorrected_SVRfun(P).reshape(n, n, n)
    ff_pureNet = net_fun(P).reshape(n, n, n)
    ff_pureNetNN = net_fun_NN(P).reshape(n, n, n)




    iso_val = 0.0
    verts_VM, faces_VM, _, _ = marching_cubes(ff_VM, iso_val, spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))
    verts_fCB, faces_fCB, _, _ = marching_cubes(ff_fCB, iso_val, spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))
    verts_cGP, faces_cGP, _, _ = marching_cubes(ff_VMCorrectedGP, iso_val,
                                                spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))
    verts_cSVR, faces_cSVR, _, _ = marching_cubes(ff_VMCorrectedSVR, iso_val,
                                                  spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))
    verts_cNN, faces_cNN, _, _ = marching_cubes(ff_VMCorrected, iso_val,
                                                spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))
    verts_pureNet, faces_pureNet, _, _ = marching_cubes(ff_pureNet, iso_val,
                                                        spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))
    verts_pureNetNN, faces_pureNetNN, _, _ = marching_cubes(ff_pureNetNN, iso_val,
                                                            spacing=(4 * sigv / n, 4 * sigv / n, 4 * sigv / n))


    verts_VM = verts_VM - 2 * sigv
    verts_fCB = verts_fCB - 2 * sigv
    verts_cGP = verts_cGP - 2 * sigv
    verts_cSVR = verts_cSVR - 2 * sigv
    verts_pureNet = verts_pureNet - 2 * sigv
    verts_cNN = verts_cNN - 2 * sigv
    verts_pureNetNN = verts_pureNetNN - 2 * sigv


    plt.close()
    fig = plt.figure(figsize=(7, 7), frameon=True)
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111, projection='3d')
    tris = ax.plot_trisurf(verts_VM[:, 1], verts_VM[:, 0], faces_VM, verts_VM[:, 2], color='k',
                           lw=1, alpha=0.007, label=r'model  $f_{mod}$', edgecolor=(0.0, 0.0, 0.0, 0.0005))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    tris = ax.plot_trisurf(verts_fCB[:, 1], verts_fCB[:, 0], faces_fCB, verts_fCB[:, 2], color=(0.0, 0.9, 0.3),
                           lw=1, alpha=0.02, label=r'true $f$', edgecolor=(0.0, 0.9, 0.3, 0.001))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    for i in np.arange(verts_pureNetNN.shape[0]):
        if verts_pureNetNN[i, 0] > 600.:
            verts_pureNetNN[i, 0] = np.nan
        if verts_pureNetNN[i, 0] < -600.:
            verts_pureNetNN[i, 0] = np.nan
        if verts_pureNetNN[i, 1] > 600.:
            verts_pureNetNN[i, 1] = np.nan
        if verts_pureNetNN[i, 1] < -600.:
            verts_pureNetNN[i, 1] = np.nan
        if verts_pureNetNN[i, 2] > 600.:
            verts_pureNetNN[i, 2] = np.nan
        if verts_pureNetNN[i, 2] < -600.:
            verts_pureNetNN[i, 2] = np.nan
        else:
            pass

    tris = ax.plot_trisurf(verts_pureNetNN[:, 1], verts_pureNetNN[:, 0], faces_pureNetNN, verts_pureNetNN[:, 2],
                           color='b', linewidth=1.0, alpha=0.03, edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'fitted $f$')
    # Create a Rectangle patch
    # ax.plot([-1110,-10000], [-1110,-10001],color='b',label=r'corrected $f$')
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    ax.scatter(inp_data[:, 0], inp_data[:, 1], inp_data[:, 2], c='k', s=10)
    plt.xlim([-600, 600])
    plt.ylim([-600, 600])
    ax.set_zlim(-600, 600)
    ax.set_xlabel(r'$\sigma_{xx}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{yy}$', fontsize=18)
    ax.set_zlabel(r'$\sigma_{xy}$', fontsize=18)
    leg = plt.legend(fontsize=14, loc='upper left', ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.view_init(elev=20., azim=-40)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    ax.grid(False)


    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = pylab.array([xlims[0], ylims[0], zlims[0]])
    f = pylab.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(pylab.array([[i, f]]))
    p.set_color((0, 0, 0, 0.1))
    # p.set_alpha(0.2)
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # fig.tight_layout()
    # ax.spines[item].set_linewidth( 1 )
    ST = 'mDataDriven/Images/FullStressPureNN_NNNew_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()

    plt.close()
    fig = plt.figure(figsize=(7, 7), frameon=True)
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111, projection='3d')
    tris = ax.plot_trisurf(verts_VM[:, 1], verts_VM[:, 0], faces_VM, verts_VM[:, 2], color='k',
                           lw=1, alpha=0.007, label=r'model  $f_{mod}$', edgecolor=(0.0, 0.0, 0.0, 0.0005))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    tris = ax.plot_trisurf(verts_fCB[:, 1], verts_fCB[:, 0], faces_fCB, verts_fCB[:, 2], color=(0.0, 0.9, 0.3),
                           lw=1, alpha=0.02, label=r'true $f$', edgecolor=(0.0, 0.9, 0.3, 0.001))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    tris = ax.plot_trisurf(verts_cSVR[:, 1], verts_cSVR[:, 0], faces_cSVR, verts_cSVR[:, 2], color='b', linewidth=1.0,
                           alpha=0.03, edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'corrected $f$')
    # Create a Rectangle patch
    # ax.plot([-1110,-10000], [-1110,-10001],color='b',label=r'corrected $f$')
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    ax.scatter(inp_data[:, 0], inp_data[:, 1], inp_data[:, 2], c='k', s=10)

    plt.xlim([-600, 600])
    plt.ylim([-600, 600])
    ax.set_zlim(-600, 600)
    ax.set_xlabel(r'$\sigma_{xx}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{yy}$', fontsize=18)
    ax.set_zlabel(r'$\sigma_{xy}$', fontsize=18)

    leg = plt.legend(fontsize=14, loc='upper left', ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.view_init(elev=20., azim=-40)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    ax.grid(False)



    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = pylab.array([xlims[0], ylims[0], zlims[0]])
    f = pylab.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(pylab.array([[i, f]]))
    p.set_color((0, 0, 0, 0.1))
    # p.set_alpha(0.2)
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # fig.tight_layout()
    # ax.spines[item].set_linewidth( 1 )
    ST = 'mDataDriven/Images/FullStressCorrectedSVRNew_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()

    plt.close()
    fig = plt.figure(figsize=(7, 7), frameon=True)
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111, projection='3d')
    tris = ax.plot_trisurf(verts_VM[:, 1], verts_VM[:, 0], faces_VM, verts_VM[:, 2], color='k',
                           lw=1, alpha=0.007, label=r'model  $f_{mod}$', edgecolor=(0.0, 0.0, 0.0, 0.0005))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    tris = ax.plot_trisurf(verts_fCB[:, 1], verts_fCB[:, 0], faces_fCB, verts_fCB[:, 2], color=(0.0, 0.9, 0.3),
                           lw=1, alpha=0.02, label=r'true $f$', edgecolor=(0.0, 0.9, 0.3, 0.001))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    tris = ax.plot_trisurf(verts_pureNet[:, 1], verts_pureNet[:, 0], faces_pureNet, verts_pureNet[:, 2], color='b',
                           linewidth=1.0, alpha=0.03, edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'fitted $f$')
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    ax.scatter(inp_data[:, 0], inp_data[:, 1], inp_data[:, 2], c='k', s=10)

    plt.xlim([-600, 600])
    plt.ylim([-600, 600])
    ax.set_zlim(-600, 600)
    ax.set_xlabel(r'$\sigma_{xx}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{yy}$', fontsize=18)
    ax.set_zlabel(r'$\sigma_{xy}$', fontsize=18)

    leg = plt.legend(fontsize=14, loc='upper left', ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.view_init(elev=20., azim=-40)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    ax.grid(False)



    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = pylab.array([xlims[0], ylims[0], zlims[0]])
    f = pylab.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(pylab.array([[i, f]]))
    p.set_color((0, 0, 0, 0.1))
    # p.set_alpha(0.2)
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # fig.tight_layout()
    # ax.spines[item].set_linewidth( 1 )
    ST = 'mDataDriven/Images/FullStressPureNNNew_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()

    plt.close()
    fig = plt.figure(figsize=(7, 7), frameon=True)
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111, projection='3d')
    tris = ax.plot_trisurf(verts_VM[:, 1], verts_VM[:, 0], faces_VM, verts_VM[:, 2], color='k',
                           lw=1, alpha=0.007, label=r'model  $f_{mod}$', edgecolor=(0.0, 0.0, 0.0, 0.0005))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    tris = ax.plot_trisurf(verts_fCB[:, 1], verts_fCB[:, 0], faces_fCB, verts_fCB[:, 2], color=(0.0, 0.9, 0.3),
                           lw=1, alpha=0.02, label=r'true $f$', edgecolor=(0.0, 0.9, 0.3, 0.001))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    ax.scatter(inp_data[:, 0], inp_data[:, 1], inp_data[:, 2], c='k', s=10)

    plt.xlim([-600, 600])
    plt.ylim([-600, 600])
    ax.set_zlim(-600, 600)
    ax.set_xlabel(r'$\sigma_{xx}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{yy}$', fontsize=18)
    ax.set_zlabel(r'$\sigma_{xy}$', fontsize=18)

    leg = plt.legend(fontsize=14, loc='upper left', ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.view_init(elev=20., azim=-40)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    ax.grid(False)



    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = pylab.array([xlims[0], ylims[0], zlims[0]])
    f = pylab.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(pylab.array([[i, f]]))
    p.set_color((0, 0, 0, 0.1))
    # p.set_alpha(0.2)
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # fig.tight_layout()
    # ax.spines[item].set_linewidth( 1 )
    ST = 'mDataDriven/Images/FullStressPure_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()

    plt.close()
    fig = plt.figure(figsize=(7, 7), frameon=True)
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111, projection='3d')
    tris = ax.plot_trisurf(verts_VM[:, 1], verts_VM[:, 0], faces_VM, verts_VM[:, 2], color='k',
                           lw=1, alpha=0.007, label=r'model  $f_{mod}$', edgecolor=(0.0, 0.0, 0.0, 0.0005))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    tris = ax.plot_trisurf(verts_fCB[:, 1], verts_fCB[:, 0], faces_fCB, verts_fCB[:, 2], color=(0.0, 0.9, 0.3),
                           lw=1, alpha=0.02, label=r'true $f$', edgecolor=(0.0, 0.9, 0.3, 0.001))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    # tris = ax.plot_trisurf(verts_pureNet[:, 0], verts_pureNet[:,1], faces_pureNet, verts_pureNet[:, 2], color='b', linewidth = 1.0   ,alpha=0.03,edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'fitted $f$')
    tris = ax.plot_trisurf(verts_cGP[:, 1], verts_cGP[:, 0], faces_cGP, verts_cGP[:, 2], color='b', linewidth=1.0,
                           alpha=0.03, edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'corrected $f$')
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    ax.scatter(inp_data[:, 0], inp_data[:, 1], inp_data[:, 2], c='k', s=10)

    plt.xlim([-600, 600])
    plt.ylim([-600, 600])
    ax.set_zlim(-600, 600)
    ax.set_xlabel(r'$\sigma_{xx}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{yy}$', fontsize=18)
    ax.set_zlabel(r'$\sigma_{xy}$', fontsize=18)

    leg = plt.legend(fontsize=14, loc='upper left', ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.view_init(elev=20., azim=-40)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    ax.grid(False)



    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = pylab.array([xlims[0], ylims[0], zlims[0]])
    f = pylab.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(pylab.array([[i, f]]))
    p.set_color((0, 0, 0, 0.1))
    # p.set_alpha(0.2)
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # fig.tight_layout()
    # ax.spines[item].set_linewidth( 1 )
    ST = 'mDataDriven/Images/FullStressGPNew_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()

    plt.close()
    fig = plt.figure(figsize=(7, 7), frameon=True)
    plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111, projection='3d')
    tris = ax.plot_trisurf(verts_VM[:, 1], verts_VM[:, 0], faces_VM, verts_VM[:, 2], color='k',
                           lw=1, alpha=0.007, label=r'model  $f_{mod}$', edgecolor=(0.0, 0.0, 0.0, 0.0005))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d
    tris = ax.plot_trisurf(verts_fCB[:, 1], verts_fCB[:, 0], faces_fCB, verts_fCB[:, 2], color=(0.0, 0.9, 0.3),
                           lw=1, alpha=0.02, label=r'true $f$', edgecolor=(0.0, 0.9, 0.3, 0.001))
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    # tris = ax.plot_trisurf(verts_pureNet[:, 0], verts_pureNet[:,1], faces_pureNet, verts_pureNet[:, 2], color='b', linewidth = 1.0   ,alpha=0.03,edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'fitted $f$')
    # tris = ax.plot_trisurf(verts_cGP[:, 0], verts_cGP[:,1], faces_cGP, verts_cGP[:, 2], color='b', linewidth = 1.0   ,alpha=0.03,edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'corrected $f$')
    tris = ax.plot_trisurf(verts_cNN[:, 1], verts_cNN[:, 0], faces_cNN, verts_cNN[:, 2], color='b', linewidth=1.0,
                           alpha=0.03, edgecolor=(0.1, 0.2, 0.5, 0.001), label=r'corrected $f$')
    tris._facecolors2d = tris._facecolors3d
    tris._edgecolors2d = tris._edgecolors3d

    ax.scatter(inp_data[:, 0], inp_data[:, 1], inp_data[:, 2], c='k', s=10)

    plt.xlim([-600, 600])
    plt.ylim([-600, 600])
    ax.set_zlim(-600, 600)
    ax.set_xlabel(r'$\sigma_{xx}$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{yy}$', fontsize=18)
    ax.set_zlabel(r'$\sigma_{xy}$', fontsize=18)

    leg = plt.legend(fontsize=14, loc='upper left', ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.view_init(elev=20., azim=-40)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    ax.grid(False)



    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = pylab.array([xlims[0], ylims[0], zlims[0]])
    f = pylab.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(pylab.array([[i, f]]))
    p.set_color((0, 0, 0, 0.1))
    # p.set_alpha(0.2)
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # fig.tight_layout()
    # ax.spines[item].set_linewidth( 1 )
    ST = 'mDataDriven/Images/FullStressNNNew_' + str(number) + '.pdf'
    plt.savefig(ST)
    # plt.show()



def fCB(sigx,sigy,sigxy):
    a1 = 0.869
    a2 = 3.371
    a3 = 3.509
    a4 = 1
    a5 = 0.
    a6 = 0.

    b1 = -1.591
    b2 = 5.414
    b3 = 3.957
    b4 = 0.259
    b5 = 1
    b10 = 1
    b6 = 0.
    b7 = 0.
    b8 = 0.
    b9 = 0.
    cv = 2.01
    tau = 300

    J2 = (a1 / 6) * (sigx - sigy) ** 2 + (a2 / 6) * (sigy - sigz) ** 2 + (a3 / 6) * (sigx - sigz) ** 2 + a4 * (
        sigxy) ** 2 + a5 * (sigxz) ** 2 + a6 * (sigyz) ** 2
    J3 = ((b1 + b2) / 27) * (sigx) ** 3 + ((b3 + b4) / 27) * sigy ** 3 + (1 / 27) * (2 * (b1 + b4) - b2 - b3) * (
        sigz) ** 3 - (1 / 9) * (b1 * sigy + b2 * sigz) * (sigx) ** 2 - (1 / 9) * (b3 * sigz + b4 * sigx) * (
                     sigy ** 2) - (1 / 9) * ((b1 - b2 + b4) * sigx + (b1 - b3 + b4) * sigy) * (sigz ** 2) + (2 / 9) * (
                     b1 + b4) * sigx * sigy * sigz - ((sigxz ** 2) / 3) * (
                     2 * b9 * sigy - b8 * sigz - (2 * b9 - b8) * sigx) - ((sigxy ** 2) / 3) * (
                     2 * b10 * sigz - b5 * sigy - (2 * b10 - b5) * sigx) - ((sigyz ** 2) / 3) * (
                     (b6 + b7) * sigx - b6 * sigy - b7 * sigz)
    fCB = J2 ** (3 / 2) - cv * J3 - tau ** 3
    return fCB


def fVM(sigx,sigy,sigxy):
    F = 0.433973
    G = 0.552709
    H = 0.447291
    N = 0.472479
    L = 0.
    M = 0.
    sigz = 0.
    sigxz = 0.
    sigyz = 0.
    sig = torch.tensor(430.384,requires_grad=False)
    fVM2 = F * torch.pow(sigy - sigz,2) + G * torch.pow(sigz - sigx,2) + H * torch.pow(sigx - sigy,2) + 2 * N * torch.pow(sigxy,2) - torch.pow(sig,2)
    return fVM2


def fVM_Corrected(sigx,sigy,sigxy):
    F = 0.433973
    G = 0.552709
    H = 0.447291
    N = 0.472479
    L = 0.
    M = 0.
    sigz = 0.
    sigxz = 0.
    sigyz = 0.
    sig = torch.tensor(430.384,requires_grad=False)
    fVM2 = F * torch.pow(sigy - sigz,2) + G * torch.pow(sigz - sigx,2) + H * torch.pow(sigx - sigy,2) + 2 * N * torch.pow(sigxy,2) - torch.pow(sig,2)

    inp = torch.hstack((sigx,sigy,sigxy)).reshape(1, -1)

    inp_scaled = (inp - x_min_torch) / (x_max_torch- x_min_torch)
    pred_norm = icnn_corrected(inp_scaled)

    pred = pred_norm * (y_max_torch - y_min_torch) + y_min_torch

    fVM_corrected = fVM2+ pred[0][0]

    return fVM_corrected




def getStress(ep11,ep22,ep12,ev11,ev22,ev12):
    sig11 = inv_S[0,0]*(ev11-ep11) + inv_S[0,1]*(ev22-ep22)+ inv_S[0,2]*(ev12-ep12)
    sig22 = inv_S[1,0]*(ev11-ep11) + inv_S[1,1]*(ev22-ep22)+ inv_S[1,2]*(ev12-ep12)
    sig12 = inv_S[2,0]*(ev11-ep11) + inv_S[2,1]*(ev22-ep22)+ inv_S[2,2]*(ev12-ep12)

    return (sig11,sig22,sig12)

def getDevStress_new(sig11,sig12,sig22):
    dim= 2

    tracesig = sig11+sig22
    sig_m = (1/dim)*tracesig
    devsig11 = sig11 - sig_m
    devsig12 = sig12
    devsig22 = sig22 - sig_m
    return  devsig11,devsig12,devsig22

def getJ(devsig11,devsig12,devsig22):
    a = 0.5*(devsig11**2   + 2*devsig12**2 + devsig22**2 )
    return a
def phi_J_new(sig11,sig12,sig22):
    devsnp1 = getDevStress_new(sig11,sig12,sig22)
    sig = torch.tensor(430.384,requires_grad=False)
    j2 = getJ(devsnp1[0],devsnp1[1],devsnp1[2])
    return torch.sqrt(3*j2)-sig

def numSolveFelam(xp,etrin,ep_old,lam_old,acc):


    enp_old =  ((ep_old[0], ep_old[2]),
            (ep_old[2], ep_old[1]))


    enp1 = ((xp[0], xp[2]),
            (xp[2], xp[1]))

    snp1 = getStress(xp[0],xp[1],xp[2],etrin[0],etrin[1],etrin[2])

    f_y =f_fun(snp1[0],snp1[1],snp1[2])
    dffsig = jacobian(f_fun,(snp1[0], snp1[1], snp1[2]))

    dffsig_mat = ((dffsig[0], dffsig[2]),
                (dffsig[2], dffsig[1]))

    Res11 =enp1[0][0] - enp_old[0][0] -(xp[3]-lam_old)*dffsig_mat[0][0]
    Res22 =enp1[1][1] - enp_old[1][1] -(xp[3]-lam_old)*dffsig_mat[1][1]
    Res12 =enp1[0][1] - enp_old[0][1] -(xp[3]-lam_old)*dffsig_mat[0][1]

    res = f_y
    res2 = xp[4]-acc - (xp[3]-lam_old)

    return (Res11,Res22,Res12,res,res2)



def run(ex_end,ey_end,exy_end):
    Ne = 201
    exvec = torch.linspace(0.,ex_end,Ne)
    dele = exvec[1]-exvec[0]

    eyvec = torch.as_tensor([(j)*(ey_end/(Ne-1)) for j in range(Ne)])
    exyvec =torch.as_tensor([(j)*(exy_end/(Ne-1)) for j in range(Ne)])

    evec = torch.vstack((exvec,eyvec,exyvec))
    epvec = torch.zeros_like(evec)
    sigvec= torch.zeros_like(evec)
    lam_vec = torch.zeros_like(exvec)
    f_vec = torch.zeros_like(exvec)


    acc_e = torch.tensor(0.0)

    for i in range(Ne):
        sig11,sig22,sig12 = getStress(epvec[0,i],epvec[1,i],epvec[2,i],evec[0,i], evec[1,i],evec[2,i])

        f = f_fun(sig11,sig22,sig12)
        accnp1_e_tr = acc_e
        if (f< 0):
            sigvec[0,i] = sig11
            sigvec[1,i] = sig22
            sigvec[2,i] = sig12
            f_vec[i] = f


            accmp1_e = accnp1_e_tr
        else:

            x = torch.tensor(
                [(epvec[0, i-1]), (epvec[1, i-1]), (epvec[2, i-1]),(lam_vec[i-1]),(accnp1_e_tr)], requires_grad=True).float()
            lam_old =lam_vec[i-1].clone().detach().requires_grad_(False)
            ev = torch.tensor(
                [(evec[0, i]), (evec[1, i]), (evec[2, i])], requires_grad=False)
            acc =accnp1_e_tr.clone().detach().requires_grad_(False)
            enp_old = torch.tensor(
                [(epvec[0, i-1]), (epvec[1, i-1]), (epvec[2, i-1])], requires_grad=False)

            iter = 0

            for ii in range(12):

                Q = numSolveFelam(x, ev, enp_old,lam_old,acc)
                J = jacobian(numSolveFelam, (x, ev, enp_old,lam_old,acc))
                Q = torch.stack(list(Q), dim=0)
                J = (J[0][0], J[1][0], J[2][0], J[3][0], J[4][0])
                J = torch.stack(list(J), dim=0)
                x = x -1.* torch.matmul(torch.inverse(J), Q)
                ff = torch.linalg.norm(Q).cpu().detach().numpy()
                if ii <1:
                    ff0 = ff

                iter = iter + 1

            epvec[0,i] = x[0]
            epvec[1, i] = x[1]
            epvec[2, i] = x[2]
            lam_vec[i] = x[3]

            sig11, sig22, sig12 = getStress(epvec[0, i], epvec[1, i], epvec[2, i], evec[0, i], evec[1, i], evec[2, i])
            sigvec[0,i] = sig11
            sigvec[1,i] = sig22
            sigvec[2,i] = sig12
            accmp1_e = x[4]
            f = f_fun(sig11, sig22, sig12)
            f_vec[i] = f

        acc_e = accmp1_e


    ev_np = evec.cpu().detach().numpy()
    ep_np = epvec.cpu().detach().numpy()
    sigvec_np = sigvec.cpu().detach().numpy()
    f_vec_np = f_vec.cpu().detach().numpy()
    return (ev_np,ep_np,sigvec_np,f_vec_np)


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------
    ## Load 28 training data points obtained from uniaxial and biaxial tests
    # ----------------------------------------------------------------------------------
    unidata = np.load('mDataDriven/Data/uni_data.npz',allow_pickle=True)
    unidata_in = unidata['uni_data_in'].astype(float)
    unidata_out = unidata['uni_data_out'].astype(float)

    bidata = np.load('mDataDriven/Data/bi_data.npz',allow_pickle=True)
    bidata_in = bidata['bi_data_in'].astype(float)
    bidata_out = bidata['bi_data_out'].astype(float)

    distance, index = spatial.KDTree(bidata_in).query(unidata_in)
    idx = np.where(distance>1e-2)[0]
    unidata_in=unidata_in[idx,:]
    unidata_out = unidata_out[idx,:]


    inp_data = np.vstack((bidata_in,unidata_in))
    out_data = np.vstack((bidata_out,unidata_out))
    out_data_zeros = np.zeros_like(out_data)

    number = inp_data.shape[0]

    # ----------------------------------------------------------------------------------
    ## Data preprocessing
    # ----------------------------------------------------------------------------------
    addPoints = []
    mm =0
    for i in range(mm):
        x0 = np.random.uniform(-2*sigv, 2*sigv,3)
        distance, index = spatial.KDTree(inp_data).query(x0)
        if distance>200:
            addPoints.append(x0)

    addPoints = np.asarray(addPoints)
    if mm==0:
        inp_data_added = inp_data
        out_data_added = out_data
    else:
        inp_data_added = np.vstack((inp_data,addPoints))
        out_data_added = np.vstack((out_data,np.zeros((addPoints.shape[0],1))))



    # ----------------------------------------------------------------------------------
    ## Establish GP correction model
    # ----------------------------------------------------------------------------------
    file_GP_model_Corr = 'mDataDriven/Models/GP_model_corr_'+str(number)+'Points.pkl'
    with open(file_GP_model_Corr, 'rb') as inp:
        gp_corrected = pickle.load(inp)


    # ----------------------------------------------------------------------------------
    ## Establish ICNN correction model
    # ----------------------------------------------------------------------------------
    icnn_corrected = icnn.ICNN_net(inp=3, out=1, activation=torch.nn.ReLU(), num_hidden_units=30, num_layers=3)
    icnn_corrected.recover_model('mDataDriven/Models/ICNN_model_corr_'+str(number)+'Points.pt')

    # ----------------------------------------------------------------------------------
    ## Establish ICNN model trained on zeros without correction
    # ----------------------------------------------------------------------------------
    net_pure = icnn.ICNN_net(inp=3, out=1, activation=torch.nn.ReLU(), num_hidden_units=30, num_layers=3)
    net_pure.recover_model('mDataDriven/Models/ICNN_model_'+str(number)+'Points.pt')

    # ----------------------------------------------------------------------------------
    ## Establish NN model trained on zeros without correction
    # ----------------------------------------------------------------------------------
    net_pureNN = Nnn.NN_net(inp=3, out=1, activation=torch.nn.ReLU(), num_hidden_units=30, num_layers=3)
    net_pureNN.recover_model('mDataDriven/Models/NN_model_'+str(number)+'Points.pt')

    # ----------------------------------------------------------------------------------
    ## Establish SVR correction model
    # ----------------------------------------------------------------------------------
    with open('mDataDriven/Models/SVR_model_corr_'+str(number)+'Points.pkl', 'rb') as inp:
        svr_corrected = pickle.load(inp)


    # ----------------------------------------------------------------------------------
    ## Get scalers
    # ----------------------------------------------------------------------------------
    x_scaler = MinMaxScaler()
    x_norm = x_scaler.fit_transform(inp_data)

    x_scaler_added = MinMaxScaler()
    x_norm_added = x_scaler_added.fit_transform(inp_data_added)

    y_scaler = MinMaxScaler()
    y_norm = y_scaler.fit_transform(out_data)

    y_scaler_added = MinMaxScaler()
    y_norm_added = y_scaler_added.fit_transform(out_data_added)

    x_min_torch = torch.as_tensor(x_scaler_added.data_min_).float()
    x_max_torch = torch.as_tensor(x_scaler_added.data_max_).float()
    y_min_torch = torch.as_tensor(y_scaler_added.data_min_).float()
    y_max_torch = torch.as_tensor(y_scaler_added.data_max_).float()


    inp = inp_data[0,:].reshape(1, 3)
    inp_scaled = x_scaler.transform(inp)
    inp_Test_norm = torch.as_tensor(inp_scaled).float()


    ## For data point plotting
    betavec0_true = np.linspace(0.0,180,400)
    betavec0_plot = np.array((0., 30., 45, 60., 90., 120., 135., 150.))
    #betavec0_plot = np.array((0., 45, 90., 120., 150.))
    #betavec0_plot = np.array((0., 45., 90.))

    # ----------------------------------------------------------------------------------
    ## Obtain uniaxial plots
    # ----------------------------------------------------------------------------------
    print('----------------------------------------------------------------------------------')
    print('Part 1/4:  Obtain uniaxial plots')
    print('----------------------------------------------------------------------------------')
    run_uni()

    # ----------------------------------------------------------------------------------
    ## Obtain biaxial plots
    # ----------------------------------------------------------------------------------
    print('----------------------------------------------------------------------------------')
    print('Part 2/4:  Obtain biaxial plots')
    print('----------------------------------------------------------------------------------')
    run_biaxial()

    # ----------------------------------------------------------------------------------
    ## Obtain 3D plots
    # ----------------------------------------------------------------------------------
    print('----------------------------------------------------------------------------------')
    print('Part 3/4:  Obtain 3D plots')
    print('----------------------------------------------------------------------------------')
    run_3d()

    # ----------------------------------------------------------------------------------
    ## Run loading curves
    # ----------------------------------------------------------------------------------
    print('----------------------------------------------------------------------------------')
    print('Part 4/4:  Run loading curves...')


    # Loading with ICNN correction model
    print('...ICNN')
    f_fun = fVM_Corrected
    (ev_np_modelCorrect1, ep_np_modelCorrect1, sigvec_np_modelCorrect1, f_vec_np_modelCorrect1) = run(0.,-0.1,0.)
    (ev_np_modelCorrect2, ep_np_modelCorrect2, sigvec_np_modelCorrect2, f_vec_np_modelCorrect2) = run(0.0,0.1,0.)

    # Loading true model
    print('...true model')
    f_fun = fCB
    (ev_np_true1, ep_np_true1, sigvec_np_true1, f_vec_np_true1) = run(0., -0.1, 0.)
    (ev_np_true2, ep_np_true2, sigvec_np_true2, f_vec_np_true2) = run(0., 0.1, 0.)

    # Loading base model
    print('...base model')
    f_fun = fVM
    (ev_np_model1, ep_np_model1, sigvec_np_model1, f_vec_np_model1) = run(0.0, -0.1, 0.)
    (ev_np_model2, ep_np_model2, sigvec_np_model2, f_vec_np_model2) = run(0.0, 0.1, 0.)

    print('----------------------------------------------------------------------------------')

    plt.close()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(ev_np_model1[1, :], sigvec_np_model1[1, :], label=r'model $f_{mod}$', color='k', lw=2)
    plt.plot(ev_np_model2[1, :], sigvec_np_model2[1, :], color='k', lw=2)
    plt.plot(ev_np_true1[1, :], sigvec_np_true1[1, :], linestyle='--', label=r'true $f$', color='g', lw=3)
    plt.plot(ev_np_true2[1, :], sigvec_np_true2[1, :], linestyle='--', color='g', lw=3)
    plt.plot(ev_np_modelCorrect1[1, :], sigvec_np_modelCorrect1[1, :], label=r'corrected $f$', color='b', lw=3,
             alpha=0.5)
    plt.plot(ev_np_modelCorrect2[1, :], sigvec_np_modelCorrect2[1, :], color='b', lw=3, alpha=0.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.axhline(color='k', lw=1.0, ls='-')
    ax.axvline(color='k', lw=1.0, ls='-')
    ax.set_xlabel(r'Total strain $\epsilon_{y}$', fontsize=18)
    ax.set_ylabel(r'Cauchy stress $\sigma_{yy}$', fontsize=18)
    plt.grid(color='0.95', linewidth=1.5)
    plt.legend(fontsize=14, loc='upper left')
    plt.savefig('mDataDriven/Images/StressStrainEpsY_INET_' + str(number) + '.pdf')
    plt.close()
    #plt.show()

