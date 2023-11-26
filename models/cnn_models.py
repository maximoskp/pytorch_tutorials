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

# encoders

class MNISTCNNEncoder(nn.Module):
    def __init__(self):
        super(MNISTCNNEncoder, self).__init__()
        # Increasing the channels, increases the accuracy "more quickly" than
        # the model size is increased. This increase can be fine-tuned separately
        # for the encoder and the decoder. This is the advantage of CNNs in
        # comparison to FF when it comes to image processing.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2) # 28x28 -> 12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=2) # 12x12 -> 4x4
        self.relu = nn.ReLU()
        self.output_size = 1024 # 64x4x4 - to inform the main AE
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, x):
        x = self.conv1( x.to(self.device) )
        x = self.relu( x )
        x = self.conv2( x )
        x = self.relu( x )
        return x.reshape( -1, self.output_size ) # 64x4x4 = 1024
    # end forward
# end class MNISTCNNEncoder

class MNISTCNNVAEEncoder(nn.Module):
    def __init__(self, latent_size=50):
        super(MNISTCNNVAEEncoder, self).__init__()
        # Increasing the channels, increases the accuracy "more quickly" than
        # the model size is increased. This increase can be fine-tuned separately
        # for the encoder and the decoder. This is the advantage of CNNs in
        # comparison to FF when it comes to image processing.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2) # 28x28 -> 12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=2) # 12x12 -> 4x4
        self.relu = nn.ReLU()
        self.output_size = 1024 # 64x4x4 - to inform the main AE
        self.mu = nn.Linear(self.output_size, latent_size)
        self.log_var = nn.Linear(self.output_size, latent_size)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, x):
        x = self.conv1( x.to(self.device) )
        x = self.relu( x )
        x = self.conv2( x )
        x = self.relu( x )
        x.reshape( -1, self.output_size ) # 64x4x4 = 1024
        return self.relu( self.mu(x) ) , self.relu(self.log_var(x))
    # end forward
# end class MNISTCNNVAEEncoder

# decoders

class MNISTCNNDecoder(nn.Module):
    def __init__(self, latent_size=50):
        super(MNISTCNNDecoder, self).__init__()
        self.from_latent = nn.Linear(latent_size, 1024) # 64x4x4
        self.deconv1 = nn.ConvTranspose2d( 64, 32, kernel_size=6, stride=2 )
        self.deconv2 = nn.ConvTranspose2d( 32, 1, kernel_size=6, stride=2 )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, x):
        x = self.from_latent( x.to(self.device) )
        x = x.reshape( x.shape[0], 64, 4, 4 )
        x = self.deconv1( x )
        x = self.relu( x )
        x = self.deconv2( x )
        return self.sigmoid( x )
    # end forward
# end class MNISTCNNDecoder