import torch.nn as nn
from torchsummary import summary
from torch import device, cuda

class MNISTFFClassifier(nn.Module):
    def __init__(self):
        super(MNISTFFClassifier, self).__init__()
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
# end class MNISTFFClassifier