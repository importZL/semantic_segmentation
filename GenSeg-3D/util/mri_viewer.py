import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nibabel as nib
from matplotlib.widgets import Slider
from util.util import print_timestamped
import os


class MRIViewer(object):

    def __init__(self, ax, X, dim, left, colormap=cm.get_cmap('gray')):
        # colormap.set_under('w')
        self.dim = dim
        self.ax = ax
        ax.set_title('Dimension ' + str(dim))

        self.X = X
        self.slices = X.shape[dim]
        self.ind = self.slices // 2

        ax_slider = plt.axes([left, 0.1, 0.23, 0.05], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, '', 0, int(self.slices) - 1, valinit=self.ind, valstep=1)

        curr_x = self.get_X()
        self.im = ax.imshow(curr_x,
                            cmap=colormap,
                            vmin=curr_x.min(),
                            vmax=curr_x.max())

        self.update(self.ind)
        self.slider.on_changed(self.update)

    def get_X(self):
        if self.dim == 0:
            x = self.X[self.ind, :, :]
            x = np.rot90(x, k=2)
            x = np.flip(x, axis=1)
        elif self.dim == 1:
            x = self.X[:, self.ind, :]
        else:
            x = self.X[:, :, self.ind]
        return x

    def update(self, val):
        if val is not None:
            self.ind = int(val)
        curr_x = self.get_X()
        self.im.set_data(curr_x)
        # self.im = np.rot90(self.im)
        # fig.canvas.draw_idle()

        # self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_3d(np_data, title, filename, show_plots=False):
    np_data = np.rot90(np_data, axes=(0, 2))
    np_data = np.flip(np_data, axis=[2])
    fig = plt.figure()
    fig.suptitle(title)
    ax1 = fig.add_subplot(1, 3, 1)
    fig1 = MRIViewer(ax1, np_data, 0, 0.123)
    # fig1.slider.on_changed(fig1.update)
    ax2 = fig.add_subplot(1, 3, 2)
    fig2 = MRIViewer(ax2, np_data, 1, 0.4)
    ax3 = fig.add_subplot(1, 3, 3)
    fig3 = MRIViewer(ax3, np_data, 2, 0.673)
    plt.savefig(filename, bbox_inches='tight')
    # print_timestamped("Saved in " + str(filename))
    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    image_file = "./datasets/braindata/train/t1/BraTS19_2013_0_1.nii.gz"
    niftiA = nib.load(image_file)
    niftiA_data = niftiA.get_fdata()
    image_title = os.path.basename(image_file)
    plot_3d(niftiA_data, image_title, "filename", show_plots=True)
