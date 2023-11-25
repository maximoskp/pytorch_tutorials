from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64

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

# get a batch
next_batch = next(iter(train_loader))
print('len of next_batch: ', len(next_batch))
# get the labels of all images
print('next_batch[1]: ', next_batch[1])
print('len(next_batch[1]): ', len(next_batch[1]))
# print an image of the batch
print('next_batch[0][0]: ', next_batch[0][0])
plt.imshow(next_batch[0][0][0,:,:])
plt.title('showing a ' + str(next_batch[1][0]))
plt.show()