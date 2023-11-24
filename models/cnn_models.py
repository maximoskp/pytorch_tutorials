import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from torch import device, cuda
import matplotlib.pyplot as plt
import os

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # convolutional block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # convolutional block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=34, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output = nn.Linear(34*7*7, 10)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(x.size(0), -1)
        return self.softmax( self.output(x) )
    # end forward

    def summary(self):
        summary(self, (1,28,28))
    # end summary

    def plot_model(self):
        self.model_graph = draw_graph(self, input_size=(1,1,28,28), \
            expand_nested=True, graph_name='MNIST_CNN', save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class MNISTCNN