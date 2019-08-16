from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import numpy as np
import seaborn as sns

def visualize():

    vmin = 0.1
    vmax = 10
    vlevel = 0.5
    show = False
    result_file_path = "../result/3d_surface_file.h5"
    surf_name = "test_loss"

    with h5py.File("./3d_surface_file.h5",'r') as f:

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        X, Y = np.meshgrid(x, y)
        Z = np.array(f[surf_name][:])

        # Plot 3D surface
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.plot_wireframe(X, Y, Z)
        plt.show()

        # Save 2D contours image
        plt.figure()
        CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        plt.figure()
        print(result_file_path + '_' + surf_name + '_2dcontourf' + '.pdf')
        CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
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
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(result_file_path + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')
    
        if show: plt.show()

visualize()
