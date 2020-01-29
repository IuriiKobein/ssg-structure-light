from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def slice_plot(phase1, phase2, phase):
    plt_phi1, = plt.plot(phase1[512, :], label = 'phi1')
    plt_phi1, = plt.plot(phase2[512, :], label = 'phi2')
    plt_phase = plt.plot(phase[512, :], label = 'phase')
    plt.legend(handles=[plt_phi1, plt_phi2, plt_phase])
    plt.show()

def surf_plot(data):

    x = np.arange(-5, 5, 10/data.shape[0])
    y = np.arange(-5, 5, 10/data.shape[1])
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, data, cmap='gray',
                           linewidth=0, antialiased=False)
    plt.show()