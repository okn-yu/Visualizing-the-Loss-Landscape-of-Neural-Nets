import torch
import numpy as np
import h5py
import os
from src.model.test_model import eval_loss


def calulate_loss_landscape(model, directions):
    setup_surface_file()
    init_weights = [p.data for p in model.parameters()]

    with h5py.File("./3d_surface_file.h5", 'r+') as f:
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]

        shape = (len(xcoordinates), len(ycoordinates))
        losses = np.ones(shape=shape)
        accuracies = np.ones(shape=shape)

        if not ("test_loss" in f.keys()):
            f["test_loss"] = losses
        if not ("test_acc" in f.keys()):
            f["test_acc"] = accuracies

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            print("count...%s" % count)
            coord = coords[count]
            overwrite_weights(model, init_weights, directions, coord)

            loss, acc = eval_loss(model)
            print(loss, acc)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            f["test_loss"][:] = losses
            f["test_acc"][:] = accuracies
            f.flush()

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
                count, len(inds), 100.0 * count / len(inds), str(coord)))


def setup_surface_file():
    xmin, xmax, xnum = -1, 1, 51
    ymin, ymax, ynum = -1, 1, 51

    surface_path = "./3d_surface_file.h5"

    if os.path.isfile(surface_path):
        print("%s is already set up" % "surface_file.h5")
        return

    with h5py.File(surface_path, 'a') as f:
        xcoordinates = np.linspace(xmin, xmax, xnum)
        f['xcoordinates'] = xcoordinates

        ycoordinates = np.linspace(ymin, ymax, ynum)
        f['ycoordinates'] = ycoordinates


def get_indices(vals, xcoordinates, ycoordinates):
    inds = np.array(range(vals.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]

    return inds, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0]
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + torch.Tensor(d).to(device)
