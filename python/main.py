import numpy as np

import read_preprocess as rp
import phase_unwrap as pu
import plots as plt


def wrap(x):
    while np.abs(x) >= np.pi:
        x = x - np.sign(x) * 2 * np.pi
    return x
wrap = np.vectorize(wrap)

if __name__ == '__main__':

    object_type = 'cube' 
    img_num = 6 #number of images in the pattern
    f1 = 1 #frequncy of the low-frequency pattern default: 1
    f2 = 60 #frequncy of the high-frequency pattern [3, 5, 10, 20, 30, 60]
    scale = 63 #scale for each high-frequency pattern [3, 5, 10, 21, 31, 63]
    real_scale = 0.144 #scale to convert phase to real value different for each frequncy [2.879, 1.727, 0.863, 0.432, 0.289, 0.144]

    phase1 = rp.get_phase(object_type, img_num, f1)
    #phase1 = wrap(phase1)

    phase2 = rp.get_phase(object_type, img_num, f2)
    phase2 = wrap(phase2)

    depth = pu.temporal_unwrap(phase1, phase2, scale, real_scale)

    plt.surf_plot(depth)

