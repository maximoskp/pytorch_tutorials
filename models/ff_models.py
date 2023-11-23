import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from torch import device, cuda
import matplotlib.pyplot as plt
import os

class MNISTFFSingleLayer(nn.Module):
    def __init__(self):
        super(MNISTFFSingleLayer, self).__init__()
        self.linear = nn.Linear(28*28, 10)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, image):
        a = image.view(-1, 28*28)
        return self.softmax( self.linear( a ) )
    # end forward

    def summary(self):
        summary(self, (1,28,28))
    # end summary

    def plot_model(self):
        self.model_graph = draw_graph(self, input_size=(1,28,28), \
            expand_nested=True, graph_name='MNIST_single_layer', save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model


    def visualize_weights(self):
        os.makedirs('figs', exist_ok=True)
        dir_name = 'figs/MNIST_single_layer_weights'
        os.makedirs(dir_name, exist_ok=True)
        for i in range(10):
            plt.imshow( self.linear.weight[i,:].reshape(28,28).cpu().detach().numpy() )
            plt.savefig(dir_name + '/weights_' + str(i) + '.png', dpi=150)
    # end visualize_weights
# end class MNISTFFSingleLayer

class MNISTFFTwoLayers(nn.Module):
    def __init__(self):
        super(MNISTFFTwoLayers, self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu( self.linear1( a ) )
        a = self.relu( self.linear2( a ) )
        return self.final( a )
    # end forward

    def summary(self):
        summary(self, (1,28,28))
    # end summary

    def plot_model(self):
        self.model_graph = draw_graph(self, input_size=(1,28,28), \
            expand_nested=True, graph_name='MNIST_two_layers', save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class MNISTFFTwoLayers