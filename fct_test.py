import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import misc
import scipy.fftpack

#Library DCT
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho', type=2), axis=1 , norm='ortho', type=2)

#Custom DCT
def fct(img):
    M = len(img)

    y = np.empty(2*M)
    y[:M] = img
    y[M:] = img[::-1]

    m = np.arange(0, M, 1)

    Y = scipy.fftpack.fft(y)[:M]*np.exp(-1j*np.pi*m/(2*M))# * np.exp(-1j*np.pi*n/(2*N))
    Y[0] = Y[0]/(np.sqrt(M))
    Y[1:] = Y[1:]*np.sqrt(2/M)
    return Y.real/2

def fct2(img):
    M = img.shape[0]
    N = img.shape[1]

    y = np.empty([2*M, 2*N])
    y[:M, :N] = img
    y[M:, :N] = img[::-1, :]
    y[:M, N:] = img[:, ::-1]
    y[M:, N:] = img[::-1, ::-1]

    m = np.arange(0, M, 1)
    n = np.arange(0, N, 1)
    m, n = np.meshgrid(m, n)
    Y = scipy.fftpack.fft2(y)[:M, :N] *np.exp(-1j*np.pi*(n*M+m*N)/(2*(N*M))) #* np.exp(-1j*np.pi*m/(2*M))
    Y[0,0] = Y[0,0]/(np.sqrt(M)*np.sqrt(N))
    Y[0,1:] = Y[0,1:]/np.sqrt(2)/M*2
    Y[1:,0] = Y[1:,0]/np.sqrt(2)/N*2
    Y[1:, 1:] = Y[1:, 1:]*np.sqrt(2/N)*np.sqrt(2/M)
    return Y.real/4


def ifct(img):
    M = len(img)
    m = np.arange(0, M, 1)

    Y = img * np.exp(1j*np.pi*m/(2*M))
    Y[0] = Y[0]*(np.sqrt(M))
    Y[1:] = Y[1:]*np.sqrt(2*M)

    y = Y
    y = scipy.fftpack.ifft(y).real
    z = np.empty([2*M])
    z[2*m]=y
    z[2*m+1] = y[M-1-m]
    y = y.real
    return z[:M]

def ifct2(img):
    x = []
    for i in range(img.shape[0]):
        x.append(ifct(img[i,:]))
    x = np.asarray(x)
    y = []
    for i in range(img.shape[1]):
        y.append(ifct(x[:,i]))
    return np.asarray(y).T


#Test iDCT
def my_ifct2(img):
    M = img.shape[0]
    N = img.shape[1]

    m = np.arange(0, M, 1)
    n = np.arange(0, N, 1)
    m, n = np.meshgrid(m, n)

    Y = (img) * np.exp(1j*np.pi*(n+m)/(2*(N*M)))

    Y[0, 0] = Y[0, 0] / (np.sqrt(2))
    Y[1:, 0] = Y[1:, 0]*np.sqrt(2 * M)
    Y[0, 1:] = Y[0, 1:]*np.sqrt(2 * N)
    Y[1:, 1:] = Y[1:, 1:]/np.sqrt(2*N)/np.sqrt(2*M)

    y = scipy.fftpack.ifft2(Y).real

    return y


x = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(idct2((x)))
print(my_ifct2((x)))
