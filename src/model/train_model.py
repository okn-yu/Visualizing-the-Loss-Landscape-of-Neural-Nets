import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from src.data.dataloader import dataloader

NUM_EPOCHS = 40

def prepare_trained_model(model):
    train_loader, test_loader = dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        # val
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)

        print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(NUM_EPOCHS), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()

    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(NUM_EPOCHS), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.grid()

    return model