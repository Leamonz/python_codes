import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# preparing datas
training_data = datasets.FashionMNIST(
    root="../datas",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="../datas",
    train=False,
    download=True,
    transform=ToTensor()
)
batch_size = 64

train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# for X, Y in test_loader:
#     print(f"The shape of input X: [N, C, H, W]: {X.shape}")
#     print(f"The shape of label Y: {Y.shape} {Y.dtype}")
#     break

device = "cuda" if torch.cuda.is_available() else "cpu"


# print(f"Using {device} device")


# creating models
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_network = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x.to(device)
        logits = self.linear_relu_network(x)
        return logits


model = Model().to(device)
# print(model)

# optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    # sets the module in training mode
    model.train()
    for batch_idx, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        pred = model(inputs)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Sets the module in evaluation mode
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            pred = model(inputs)
            test_loss += criterion(pred, target).item()
            pred = pred.argmax(dim=1)
            correct += (pred == target).type(torch.float32).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    for epoch in range(5):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_loader, model, criterion, optimizer)
        test(test_loader, model, criterion)
    print("Done!")
    # Saving models
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Loading models
    # classes = [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot",
    # ]
    #
    # model.eval()
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     pred = model(x)
    #     predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')
