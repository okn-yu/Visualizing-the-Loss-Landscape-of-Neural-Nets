import torch
import numpy as np
import h5py
import os
from src.model.test_model import eval_loss


def calulate(model, directions):
    setup_surface_file()
    init_weights = [p.data for p in model.parameters()]

    with h5py.File("./result/surface_file.h5", 'r+') as f:
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]

        shape = (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)

        f["train_loss"] = losses
        f["train_acc"] = accuracies

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            print("count...%s" % count)
            coord = coords[count]
            overwrite_weights(model, init_weights, directions, coord)

            loss, acc = eval_loss(model)
            print(loss, acc)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            f["train_loss"][:] = losses
            f["train_acc"][:] = accuracies
            f.flush()

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
                count, len(inds), 100.0 * count / len(inds), str(coord)))


def setup_surface_file():
    xmin, xmax, xnum = -1, 1, 51
    ymin, ymax, ynum = -1, 1, 51

    surface_path = "./result/surface_file.h5"

    if os.path.isfile(surface_path):
        print("%s is already set up" % "surface_file.h5")
        return

    with h5py.File(surface_path, 'a') as f:
        f['dir_file'] = "test_dir_name"

        xcoordinates = np.linspace(xmin, xmax, xnum)
        f['xcoordinates'] = xcoordinates

        ycoordinates = np.linspace(ymin, ymax, ynum)
        f['ycoordinates'] = ycoordinates


def get_indices(vals, xcoordinates, ycoordinates):
    """
    Args:
      vals: values at (x, y), with value -1 when the value is not yet calculated.
      xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
      ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]

    Returns:
      - a list of indices into vals for points that have not yet been calculated.
      - a list of corresponding coordinates, with one x/y coordinate per row.
    """

    # Create a list of indices into the vectorizes vals
    inds = np.array(range(vals.size))

    # Select the indices of the un-recorded entries, assuming un-recorded entries
    # will be smaller than zero. In case some vals (other than loss values) are
    # negative and those indexces will be selected again and calcualted over and over.
    inds = inds[vals.ravel() <= 0]

    # Make lists containing the x- and y-coodinates of the points to be plotted
    # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]
    return inds, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0]
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w + torch.Tensor(d).type(type(w))
