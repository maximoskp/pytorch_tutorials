from models.cnn_models import MNISTCNN
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import save, load
import os

load_saved_model = False

batch_size = 100
num_epochs = 4
lr = 0.00001

model = MNISTCNN()

# show model info
print(model)
model.summary()
model.plot_model()

if load_saved_model:
    model.load_state_dict(load('saved_models/MNIST_CNN.stdict'))
    model.eval()
else:
    # load previously downloaded data
    train_ds = MNIST(
        root='./data',
        train=True,
        transform=ToTensor(),
        download=False
    )
    test_ds = MNIST(
        root='./data',
        train=False,
        transform=ToTensor(),
        download=False
    )
    print(train_ds)
    print(test_ds)

    # create dataloaders
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset = test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    loss = CrossEntropyLoss()
    optimizer = Adam( model.parameters(), lr=lr )

    number_of_batches = len(train_loader)

    for epoch in range(num_epochs):
        for x, (images, labels) in enumerate(train_loader):
            images_reshaped = images.reshape(-1, 1, 28, 28)
            
            prediction = model( images_reshaped.to(model.device) )
            losses = loss(prediction, labels.to(model.device))

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (x+1)%100 == 0:
                print(f'Epochs [{epoch+1}/{num_epochs}] - Batch [{x+1}/{number_of_batches}] - Loss: {losses.item():.4f}')
            # end if
        # end for batch
    # end for epoch

    # save model
    os.makedirs('saved_models', exist_ok=True)
    save(model.state_dict(), 'saved_models/MNIST_CNN.stdict')
# end else
