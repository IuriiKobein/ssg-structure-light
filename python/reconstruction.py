from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
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

if __name__ == '__main__':
    imgs_path = '/home/rnd/Pictures/dragon/images'
    refs_files = [imgs_path + '/31_ref1.png',imgs_path +  '/31_ref2.png',imgs_path +  '/31_ref3.png',imgs_path +  '/31_ref4.png']
    refs = []
    for filename in refs_files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        #ref = gray - lpf(gray)
        refs.append(gray)

    images_files = [imgs_path + '/31_phase1.png',imgs_path + '/31_phase2.png', imgs_path +'/31_phase3.png', imgs_path +'/31_phase4.png']
    images = []
    for filename in images_files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray = np.asarray(img, dtype="float32")
        #ref = gray - lpf(gray)
        images.append(gray)

    images_x = []
    for i in range(len(images)):
        images_x.append(images[i]) #- q_lpf(refs[i], images[i]))

    a = sum(images_x) / 4
    filter_img = []
    for img in images_x:
        #img = img - a
        filter_img.append(img)

    #compute wrapped phase
    ref_phase = np.arctan2((refs[3] - refs[1]),(refs[0] - refs[2]))
    test_phase = np.arctan2((filter_img[3] - filter_img[1]),(filter_img[0] - filter_img[2]))

    phase1 = test_phase - ref_phase

    np.savetxt('phase1.txt', phase1, fmt='%f')
    plt.show()

    #phase1 = phase1 - np.mean(phase1)
    #plt.imshow(test_phase, cmap='gray')
    #plt.show()

    #plt.imshow(test_phase - ref_phase, cmap='gray')
    #plt.show()
    #np.save('test_data.npy', test_phase - ref_phase)

    #matplotlib.image.imsave("output.jpg", phase, cmap='gray')


    unwraped_phase = unwrap_phase(phase1)
    plt.imshow(-unwraped_phase, cmap='gray')
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    Y = np.arange(0, 1024, 1)
    X = np.arange(0, 1024, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, -unwraped_phase, cmap='gray',
                           linewidth=0, antialiased=False)
    plt.show()
