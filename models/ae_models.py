import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from torch import device, cuda, exp, rand_like
import matplotlib.pyplot as plt
import os
from .ff_models import ImageFFEncoder, ImageFFDecoder
from .cnn_models import MNISTCNNEncoder, MNISTCNNVAEEncoder, MNISTCNNDecoder

class MNISTFFAE(nn.Module):
    def __init__(self, image_size=(1,28,28)):
        super(MNISTFFAE, self).__init__()
        self.input_size = image_size
        self.encoder = ImageFFEncoder()
        self.decoder = ImageFFDecoder()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, image):
        x = self.encoder(image.to(self.device))
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

class MNISTCNNAE(nn.Module):
    def __init__(self, image_size=(1,28,28)):
        super(MNISTCNNAE, self).__init__()
        self.input_size = image_size
        self.latent_size = 50
        self.encoder = MNISTCNNEncoder()
        self.bottleneck = nn.Linear(self.encoder.output_size, self.latent_size)
        self.decoder = MNISTCNNDecoder(latent_size=self.latent_size)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, image):
        x = self.encoder(image.to(self.device))
        x = self.bottleneck(x)
        return self.decoder(x)
    # end forward
    
    def summary(self):
        summary(self, self.input_size)
    # end summary

    def plot_model(self, name='MNIST_CNN_AE'):
        self.model_graph = draw_graph(self, input_size=self.input_size, \
            expand_nested=True, graph_name=name, save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class MNISTCNNAE

class MNISTCNNVAE(nn.Module):
    def __init__(self, image_size=(1,28,28)):
        super(MNISTCNNVAE, self).__init__()
        self.input_size = image_size
        self.latent_size = 50
        self.encoder = MNISTCNNVAEEncoder()
        self.bottleneck = nn.Linear(self.encoder.output_size, self.latent_size)
        self.decoder = MNISTCNNDecoder(latent_size=self.latent_size)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def sampling(self, mu, log_var):
        std = exp( 0.5*log_var )
        eps = rand_like(std)
        return eps.mul(std).add(mu)
    # end sampling

    def forward(self, image):
        mu, log_var = self.encoder(image.to(self.device))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    # end forward
    
    def summary(self):
        summary(self, self.input_size)
    # end summary

    def plot_model(self, name='MNIST_CNN_VAE'):
        self.model_graph = draw_graph(self, input_size=self.input_size, \
            expand_nested=True, graph_name=name, save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model
# end class MNISTCNNVAE