from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

def lpf(img, freq, display_plot = False):
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
    if(display_plot):
    	display_plot(img, dft_shift, img_back, ccol)
    return img_back

def q_lpf(ref, img, display_plot = False):
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
    if(display_plot):
    	display_plot(img, fshift, img_back, ccol)
    return img_back

def get_phase(object_type, img_num, f):

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

    phase = test_phase - ref_phase
    return phase
