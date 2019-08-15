from src.model.alexnet import AlexNet
from src.model.train_model import prepare_trained_model
from src.directions import create_random_directions
import src.calc_loss

if __name__ == '__main__':

    model = AlexNet()
    directions = create_random_directions(model)

    trained_model = prepare_trained_model(model)
    trained_weight = [p.data for p in trained_model.parameters()]

    src.calc_loss.calulate(trained_model, trained_weight, directions)

