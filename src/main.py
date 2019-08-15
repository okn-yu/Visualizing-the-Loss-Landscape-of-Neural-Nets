from src.model.alexnet import AlexNet
from src.model.train import prepare_trained_model

from data.dataloader import dataloader
from src.directions import create_random_directions
import src.calc_loss
import src.h5file
import copy
import torch

if __name__ == '__main__':

    model = AlexNet()
    prepare_trained_model(model)

    directions = create_random_directions(model)

    model.load_state_dict(torch.load("./data/trained_model"))

    weight = [p.data for p in model.parameters()]
    state_dic = copy.deepcopy(model.state_dict())

    trainloader, testloader = dataloader()




    h5file.setup_surface_file()
    calc_loss.calulate(model, weight, directions, trainloader)




