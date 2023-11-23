from models.ff_models import MNISTFFClassifier
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

batch_size = 64
num_epochs = 10
lr = 0.00001

model = MNISTFFClassifier()

# show model info
print(model)
model.summary()

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
    print('epoch: ', epoch)
    for x, (images, labels) in enumerate(train_loader):
        images_reshaped = images.reshape(-1, 28*28)
        
        prediction = model( images_reshaped.to(model.device) )
        losses = loss(prediction, labels.to(model.device))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (x+1)%100 == 0:
            print(f'Epochs [{epoch+1}/{num_epochs}] - Batch [{x+1}/{number_of_batches}] - Loss: {losses.item():.4f}')