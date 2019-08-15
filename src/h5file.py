import h5py
import os
import numpy as np

xmin, xmax, xnum = -1, 1, 51
ymin, ymax, ynum = -1, 1, 51

def setup_surface_file():

    surface_path = "./result/surface_file.h5"

    if os.path.isfile(surface_path):
        print ("%s is already set up" % "surface_file.h5")
        return

    with h5py.File(surface_path,'a') as f:
        f['dir_file'] = "test_dir_name"

        xcoordinates = np.linspace(xmin, xmax, xnum)
        f['xcoordinates'] = xcoordinates

        ycoordinates = np.linspace(ymin, ymax, ynum)
        f['ycoordinates'] = ycoordinates