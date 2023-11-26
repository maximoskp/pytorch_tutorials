import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from torch import device, cuda, reshape
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO: at some point, create submodules for ff_models
# and submodules for encoders and decoders?

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

# encoders

class ImageFFEncoder(nn.Module):
    def __init__(self, image_size=(1,28,28), layer_sizes=[500, 200, 50]):
        super(ImageFFEncoder, self).__init__()
        self.num_layers = len(layer_sizes)
        self.input_size = image_size
        self.layers = []
        input_and_layer_sizes = [np.prod(self.input_size)]
        input_and_layer_sizes.extend(layer_sizes)
        self.layers = nn.ModuleList( [nn.Linear(input_and_layer_sizes[i], input_and_layer_sizes[i+1]) \
                                    for i in range(len(input_and_layer_sizes)-1)] )
        # TODO: define intermediate and final activations from input arguments
        self.intermediate_activations = nn.ReLU()
        self.final_activation = nn.ReLU()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, image):
        x = image.view(-1, np.prod(self.input_size)).to(self.device)
        for i,layer in enumerate(self.layers):
            x = layer(x)
            # make sure only intermediate layers get intermediate activations
            if i < self.num_layers-1:
                x = self.intermediate_activations(x)
        return self.final_activation(x)
    # end forward

    def summary(self):
        summary(self, self.input_size)
    # end summary

    def plot_model(self, name='Image_FF_AE_Encoder'):
        self.model_graph = draw_graph(self, input_size=self.input_size, \
            expand_nested=True, graph_name=name, save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class ImageFFEncoder

# decoders

class ImageFFDecoder(nn.Module):
    def __init__(self, latent_size=50, image_size=(1,28,28), layer_sizes=[200, 500]):
        super(ImageFFDecoder, self).__init__()
        # a layer needs to be added in the end for the image dimension
        self.num_layers = len(layer_sizes) + 1
        self.input_size = (1, latent_size)
        self.image_size = image_size
        self.layers = []
        input_and_layer_sizes = [latent_size]
        input_and_layer_sizes.extend(layer_sizes)
        input_and_layer_sizes.append(np.prod(self.image_size))
        self.layers = nn.ModuleList( [nn.Linear(input_and_layer_sizes[i], input_and_layer_sizes[i+1]) \
                                    for i in range(len(input_and_layer_sizes)-1)] )
        # TODO: define intermediate and final activations from input arguments
        self.intermediate_activations = nn.ReLU()
        self.final_activation = nn.Sigmoid()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, image):
        x = image.view(-1, np.prod(self.input_size))
        for i,layer in enumerate(self.layers):
            x = layer(x)
            # make sure only intermediate layers get intermediate activations
            if i < self.num_layers-1:
                x = self.intermediate_activations(x)
        return reshape(self.final_activation(x), [-1] + list(self.image_size) )
    # end forward

    def summary(self):
        summary(self, self.input_size)
    # end summary

    def plot_model(self, name='Image_FF_AE_Decoder'):
        self.model_graph = draw_graph(self, input_size=self.input_size, \
            expand_nested=True, graph_name=name, save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class ImageFFDecoder