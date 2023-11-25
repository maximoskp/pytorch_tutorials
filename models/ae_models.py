import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from torch import device, cuda, reshape
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO: move encoder and decoder in ff_models.py
# TODO: at some point, create submodules for ff_models
# and submodules for encoders and decoders?

class ImageFFEncoder(nn.Module):
    def __init__(self, image_size=(1,28,28), layer_sizes=[500, 200, 50]):
        super(ImageFFEncoder, self).__init__()
        self.num_layers = len(layer_sizes)
        self.input_size = image_size
        self.layers = []
        input_and_layer_sizes = [np.prod(self.input_size)].extend(layer_sizes)
        for i in range(len(input_and_layer_sizes)-1):
            self.layers.append(nn.Linear(input_and_layer_sizes[i], input_and_layer_sizes[i+1]))
        # TODO: define intermediate and final activations from input arguments
        self.intermediate_activations = nn.ReLU()
        self.final_activation = nn.ReLU()
    # end init

    def forward(self, image):
        x = image.view(-1, np.prod(self.input_size))
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

class ImageFFDecoder(nn.Module):
    def __init__(self, latent_size=50, image_size=(1,28,28), layer_sizes=[200, 500]):
        super(ImageFFDecoder, self).__init__()
        # a layer needs to be added in the end for the image dimension
        self.num_layers = len(layer_sizes) + 1
        self.input_size = (1, latent_size)
        self.image_size = image_size
        self.layers = []
        input_and_layer_sizes = [latent_size].extend(layer_sizes).append(np.prod(self.image_size))
        for i in range(len(input_and_layer_sizes)-1):
            self.layers.append(nn.Linear(input_and_layer_sizes[i], input_and_layer_sizes[i+1]))
        # TODO: define intermediate and final activations from input arguments
        self.intermediate_activations = nn.ReLU()
        self.final_activation = nn.Sigmoid()
    # end init

    def forward(self, image):
        x = image.view(-1, np.prod(self.input_size))
        for i,layer in enumerate(self.layers):
            x = layer(x)
            # make sure only intermediate layers get intermediate activations
            if i < self.num_layers-1:
                x = self.intermediate_activations(x)
        return reshape(self.final_activation(x), self.image_size)
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

class MNISTFFAE(nn.Module):
    def __init__(self, image_size=(1,28,28)):
        super(MNISTFFAE, self).__init__()
        self.input_size = image_size
        self.encoder = ImageFFEncoder()
        self.decoder = ImageFFDecoder()
    # end init

    def forward(self, image):
        x = self.encoder(x)
        return self.decoder(x)
    # end forward
    
    def summary(self):
        summary(self, self.input_size)
    # end summary

    def plot_model(self, name='MNIST_FF_AE'):
        self.model_graph = draw_graph(self, input_size=self.input_size, \
            expand_nested=True, graph_name=name, save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class MNISTFFAE