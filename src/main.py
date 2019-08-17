from src.model.alexnet import AlexNet
from src.model.resnet import *
from src.model.train_model import prepare_trained_model
from src.directions import create_random_directions
from src.calc_loss import calulate_loss_landscape

if __name__ == '__main__':
    #model = AlexNet()
    model = ResNet18()
    rand_directions = create_random_directions(model)
    trained_model = prepare_trained_model(model)
    calulate_loss_landscape(trained_model, rand_directions)

