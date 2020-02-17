from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
from scipy.fftpack import dct
from scipy.signal import medfilt
from skimage.restoration import unwrap_phase
import cv2
import numpy as np
import sys
import os
import phase_unwrap as pu

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

def lpf(img):
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
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

def imgs_load(path):
    imgs = []
    files = []
    for e in os.scandir(path):
        if e.is_file() == False:
            continue
        files.append(e.path)
    files = sorted(files)

    for f in files:
        print(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        imgs.append(gray)

    return imgs

def compute_phase(refs, objs):
    ref_phase = np.arctan2((refs[3] - refs[1]), (refs[0] - refs[2]))
    obj_phase = np.arctan2((objs[3] - objs[1]), (objs[0] - objs[2]))

    return obj_phase - ref_phase

if __name__ == '__main__':

    hf_refs_path = sys.argv[1]
    hf_objs_path = sys.argv[2]

    refs = imgs_load(hf_refs_path)
    objs = imgs_load(hf_objs_path)

    #for i in range(4):
    #    refs[i] = lpf(refs[i])
    #    objs[i] = lpf(objs[i])
    #compute wrapped phase

    phase = compute_phase(refs, objs)

    unwraped_phase = pu.skimage_unwrap(phase)
    plt.imshow(unwraped_phase, cmap='gray')
    plt.show()

    unwraped_phase = pu.iterative_unwrap(phase)
    plt.imshow(unwraped_phase, cmap='gray')
    plt.show()

    if len(sys.argv) == 3:
        sys.exit(0)

    lf_refs_path = sys.argv[3]
    lf_objs_path = sys.argv[4]

    lf_refs = imgs_load(lf_refs_path)
    lf_objs = imgs_load(lf_objs_path)

    lf_phase = compute_phase(lf_refs, lf_objs)

    unwraped_phase = pu.temporal_unwrap(lf_phase, phase, 30, 0.25 )
    plt.imshow(unwraped_phase, cmap='gray')
    plt.show()


