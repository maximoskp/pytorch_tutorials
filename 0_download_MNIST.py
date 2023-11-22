from torchvision import datasets as dts
from torchvision.transforms import ToTensor

traindt = dts.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

testdt = dts.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)