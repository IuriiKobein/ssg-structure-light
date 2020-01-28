from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import scipy
from scipy import signal
from scipy import misc
from scipy.fftpack import dct
from scipy.signal import medfilt
from skimage.restoration import unwrap_phase
import cv2
import numpy as np

def display_plot(img, spectr, img2, ccol):
    magnitude_spectrum = 20*np.log(np.absolute(spectr))
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img2, cmap = 'gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    fig, ax = plt.subplots()
    ax.plot(range(magnitude_spectrum.shape[0]), magnitude_spectrum[:, ccol])
    ax.grid()
    plt.show()

def lpf(img, freq):
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-freq:crow+freq, ccol-freq:ccol+freq] = 1
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.absolute(img_back)
    img_back = (img_back - np.min(img_back))/(np.max(img_back) - np.min(img_back))
    #display_plot(img, dft_shift, img_back, ccol)
    return img_back

def q_lpf(ref, img):
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols), np.uint8)
    dft_img = np.fft.fft2(img)
    dft_shift_img = np.fft.fftshift(dft_img)

    dft_ref = np.fft.fft2(ref)
    dft_shift_ref = np.fft.fftshift(dft_ref)

    fshift = dft_shift_img - dft_shift_ref
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.absolute(img_back)
    img_back = (img_back - np.min(img_back))/(np.max(img_back) - np.min(img_back))
    #display_plot(img, fshift, img_back, ccol)
    return img_back

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

def custom_phase_unwrap(phase):
    phase = wrap(phase)
    phi1 = ilaplacian(delta_phi(phase)) 
    k1 = np.around((phi1 - phase)*0.5/np.pi)
    phi2 = phase + 2*k1*np.pi
    for i in range(1,10):
        error = phi2 - phi1
        phie = ilaplacian(delta_phi(error)) * 4
        phi1 = phi1 + phie + np.mean(phi2) - np.mean(phi1)
        k2 = np.around((phi1 - phase)*0.5/np.pi)
        phi2 = phase + 2*k2*np.pi
        if (np.array_equal(k1, k2)):
            break
        k1 = k2
    return phi2


if __name__ == '__main__':

    object_type = 'sphere'
    img_num = 6 #number of images in the pattern
    f = 1 #frequncy of the low-frequency pattern default: 1

    refs = []
    for i in range(img_num):
        filename = f'images/{img_num}_{object_type}/{object_type}_{f}_ref{i}.png'
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        ref = gray - lpf(gray, 40)
        refs.append(ref)

    images = []
    for i in range(img_num):
        filename = f'images/{img_num}_{object_type}/{object_type}_{f}_phase{i}.png'
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        images.append(gray)

    images_x = []
    for i in range(len(images)):
        images_x.append(images[i] - q_lpf(refs[i], images[i]))

    a = sum(images_x) / img_num
    filter_img = []
    for img in images_x:
        img = img - a
        filter_img.append(img)

    alpha = np.pi/(img_num/2)

    a = np.zeros(refs[0].shape)
    b = np.zeros(refs[0].shape)

    for i in range(img_num):
        a -= refs[i] * np.sin(i*alpha)
        b += refs[i] * np.cos(i*alpha)

    ref_phase = np.arctan2(a, b)

    a = np.zeros(refs[0].shape)
    b = np.zeros(refs[0].shape)

    for i in range(img_num):
        a -= filter_img[i] * np.sin(i*alpha)
        b += filter_img[i] * np.cos(i*alpha)

    test_phase = np.arctan2(a, b)

    phase1 = test_phase - ref_phase
    phase1 = wrap(phase1)

    f = 3 #frequncy of the high-frequency pattern [3, 5, 10, 20, 30, 60]

    refs = []
    for i in range(img_num):
        filename = f'images/{img_num}_{object_type}/{object_type}_{f}_ref{i}.png'
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        ref = gray - lpf(gray, 40)
        refs.append(ref)

    
    images = []
    for i in range(img_num):
        filename = f'images/{img_num}_{object_type}/{object_type}_{f}_phase{i}.png'
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        images.append(gray)

    images_x = []
    for i in range(len(images)):
        images_x.append(images[i] - q_lpf(refs[i], images[i]))

    a = sum(images_x) / img_num
    filter_img = []
    for img in images_x:
        img = img - a
        filter_img.append(img)

    a = np.zeros(refs[0].shape)
    b = np.zeros(refs[0].shape)

    for i in range(img_num):
        a -= refs[i] * np.sin(i*alpha)
        b += refs[i] * np.cos(i*alpha)

    ref_phase = np.arctan2(a, b)

    a = np.zeros(refs[0].shape)
    b = np.zeros(refs[0].shape)

    for i in range(img_num):
        a -= filter_img[i] * np.sin(i*alpha)
        b += filter_img[i] * np.cos(i*alpha)

    test_phase = np.arctan2(a, b)

    phase2 = (test_phase - ref_phase)
    phase2 = wrap(phase2)

    kernel = np.ones((5,5),np.float32)/25
    phase1 = cv2.filter2D(phase1,-1,kernel)

    scale = 3 #scale for each high-frequency pattern [3, 5, 10, 21, 31, 63]
    k1 = np.around((scale*phase1 - phase2)*0.5/np.pi)
    phi2 = phase2 + 2*k1*np.pi

    #plt_phase, = plt.plot(phase1[512, :]*scale, label = 'phi1')
    #plt_phi1, = plt.plot(phase2[512, :], label = 'phi2')
    #plt_phi2, = plt.plot(phi2[512, :], label = 'phase')
    #plt.legend(handles=[plt_phase, plt_phi1, plt_phi2])
    #plt.show()
    
    x = np.arange(-5, 5, 10/1024)
    y = np.arange(-5, 5, 10/1024)

    x, y = np.meshgrid(x, y)

    real_scale = 2.879 #scale to convert phase to real value different for each frequncy [2.879, 1.727, 0.863, 0.432, 0.289, 0.144]
    phi2 = phi2*real_scale

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

    print('Absolute error: ', np.sum(np.abs(object3D-phi2))/1024/1024*100, '%')
    print('Relative error: ', np.sum(rel_error)/1024/1024*100, '%')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, object3D-phi2, cmap='gray',
                           linewidth=0, antialiased=False)
    plt.show()

