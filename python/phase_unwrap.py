import cv2
import numpy as np
from skimage.restoration import unwrap_phase
import scipy
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho', type=2), axis=1 , norm='ortho', type=2)

def laplacian(a):
    ca = np.zeros(a.shape)
    ca = dct2(a)
    Y = np.arange(0, a.shape[0], 1)
    X = np.arange(0, a.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = X*X +Y*Y
    ca *= Z
    output = idct2(ca)
    output = - 4 *np.pi*np.pi*output/(a.shape[0]*a.shape[1])
    return output

def ilaplacian(a):
    ca = np.zeros(a.shape)

    ca = dct2(a)
    Y = np.arange(1, a.shape[0]+1, 1)
    X = np.arange(1, a.shape[1]+1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = X*X +Y*Y
    ca /= Z
    output= idct2(ca)
    output = -(a.shape[0]*a.shape[1])*output/(4*np.pi*np.pi)
    return output

def delta_phi(a):
    a1 = np.cos(a)*laplacian(np.sin(a))
    a2 = np.sin(a)*laplacian(np.cos(a))
    return a1-a2

def wrap(x):
    while np.abs(x) >= np.pi:
        x = x - np.sign(x) * 2 * np.pi
    return x
wrap = np.vectorize(wrap)

def iterative_unwrap(phase):
    #phase = wrap(phase)
    phi1 = ilaplacian(delta_phi(phase))
    k1 = np.around((phi1 - phase)*0.5/np.pi)
    phi2 = phase + 2*k1*np.pi
    for i in range(1,10):
        error = phi2 - phi1
        phie = ilaplacian(delta_phi(error))
        phi1 = phi1 + phie + np.mean(phi2) - np.mean(phi1)
        k2 = np.around((phi1 - phase)*0.5/np.pi)
        phi2 = phase + 2*k2*np.pi
        if (np.array_equal(k1, k2)):
            break
        k1 = k2
    return phi2

def temporal_unwrap(phase1, phase2, scale, real_scale):

    kernel = np.ones((5,5),np.float32)/25
    phase1 = cv2.filter2D(phase1,-1,kernel)

    k1 = np.around((scale*phase1 - phase2)*0.5/np.pi)
    phi2 = phase2 + 2*k1*np.pi

    phi2 = phi2*real_scale
    return phi2

def skimage_unwrap(phase):
	return unwrap_phase(phase)


def compute_error(phase, object_type):
    x = np.arange(-5, 5, 10/phase.shape[0])
    y = np.arange(-5, 5, 10/phase.shape[1])

    x, y = np.meshgrid(x, y)

    object3D = np.empty(phi2.shape)

    if (object_type == 'cube'):
        for i in range(phi2.shape[0]):
            for j in range(phi2.shape[1]):
                if(np.abs(x[i, j]) <= 2.5 and np.abs(y[i, j]) <= 2.5):
                    object3D[i,j] = 2.5
                else:
                    object3D[i, j] = 0
    if (object_type == "sphere"):
        for i in range(phi2.shape[0]):
            for j in range(phi2.shape[1]):
                if((2.5*2.5 - x[i, j]*x[i, j] - y[i, j]*y[i, j]) >= 0):
                    object3D[i,j] = np.sqrt(2.5*2.5 - x[i, j]*x[i, j] - y[i, j]*y[i, j])
                else:
                    object3D[i, j] = 0


    rel_error = (object3D-phi2)/object3D
    rel_error[rel_error == np.inf] = 0
    rel_error[rel_error != rel_error] = 0
    rel_error[rel_error == -np.inf] = 0

    print('Absolute error: ', np.sum(np.abs(object3D-phi2))/phase.shape[0]/phase.shape[1]*100, '%')
    print('Relative error: ', np.sum(rel_error)/phase.shape[0]/phase.shape[1]*100, '%')
