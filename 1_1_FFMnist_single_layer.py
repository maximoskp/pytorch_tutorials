from models.ff_models import MNISTFFSingleLayer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import save, load
import os

load_saved_model = True

batch_size = 100
num_epochs = 10
lr = 0.00001

model = MNISTFFSingleLayer()

# show model info
print(model)
model.summary()
model.plot_model()

'''
1. Notice that the load_state_dict() function takes a dictionary object, 
NOT a path to a saved object. This means that you must deserialize the 
saved state_dict before you pass it to the load_state_dict() function. 
For example, you CANNOT load using model.load_state_dict(PATH).

2. If you only plan to keep the best performing model (according to 
the acquired validation loss), don’t forget that 
best_model_state = model.state_dict() returns a reference to the 
state and not its copy! You must serialize best_model_state or use 
best_model_state = deepcopy(model.state_dict()) otherwise your 
best best_model_state will keep getting updated by the subsequent 
training iterations. As a result, the final model state will be 
the state of the overfitted model.
'''

if load_saved_model:
    model.load_state_dict(load('saved_models/MNIST_single_layer.stdict'))
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
            images_reshaped = images.reshape(-1, 28*28)
            
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
    save(model.state_dict(), 'saved_models/MNIST_single_layer.stdict')
# end else

model.visualize_weights()