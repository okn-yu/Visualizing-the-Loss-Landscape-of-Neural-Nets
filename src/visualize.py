from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns

# matplotlib reference:
# http://pynote.hatenablog.com/entry/matplotlib-surface-plot
# https://qiita.com/kazetof/items/c0204f197d394458022a

def visualize():

    vmin = 0
    vmax = 100
    vlevel = 0.5
    result_file_path = "../result/3d_surface_file.h5"
    surf_name = "test_loss"

    with h5py.File("../3d_surface_file.h5",'r') as f:

        Z_LIMIT = 10

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])

        X, Y = np.meshgrid(x, y)
        Z = np.array(f[surf_name][:])
        #Z[Z > Z_LIMIT] = Z_LIMIT
        #Z = np.log(Z)  # logscale

        # 回転可能な3Dイメージを描画
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        #ax.plot_wireframe(X, Y, Z)
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
        plt.show()

        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        fig = plt.figure()
        CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        # Save 2D heatmaps image
        plt.figure()
        sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                               xticklabels=False, yticklabels=False)
        sns_plot.invert_yaxis()
        sns_plot.get_figure().savefig(result_file_path + '_' + surf_name + '_2dheat.pdf',
                                      dpi=300, bbox_inches='tight', format='pdf')

        # Save 3D surface image
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        fig.savefig(result_file_path + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

visualize()
