from models.ae_models import MNISTCNNAE
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from torch import save, load, Tensor
import os
import matplotlib.pyplot as plt

load_saved_model = False

batch_size = 100
num_epochs = 10
lr = 0.00001

model = MNISTCNNAE()

# show model info
print(model)
model.summary()
model.plot_model()

if load_saved_model:
    model.load_state_dict(load('saved_models/MNIST_AE_CNN.stdict'))
    model.eval()
    test_loader = None
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

    loss = BCELoss()
    optimizer = Adam( model.parameters(), lr=lr )

    number_of_batches = len(train_loader)

    for epoch in range(num_epochs):
        for x, (images, _) in enumerate(train_loader):
            # images_reshaped = images.reshape(-1, 28*28)
            
            prediction = model( images.to(model.device) )
            losses = loss(prediction, images.to(model.device))

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
    save(model.state_dict(), 'saved_models/MNIST_AE_CNN.stdict')
# end else

if test_loader is None:
    test_ds = MNIST(
        root='./data',
        train=False,
        transform=ToTensor(),
        download=False
    )
    test_loader = DataLoader(
        dataset = test_ds,
        batch_size=batch_size,
        shuffle=False
    )

# plot example from image from test set
y_real = next(iter(test_loader))[0][0]
y_predict = model(Tensor(y_real).to(model.device))[0]
plt.subplot(1,2,1)
plt.imshow(y_real.view(28,28))
plt.subplot(1,2,2)
plt.imshow(y_predict.view(28,28).cpu().detach().numpy())
plt.show()