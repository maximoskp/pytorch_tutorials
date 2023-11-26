# pytorch_tutorials
Simple tutorials on basics of pytorch.

MNIST linear and conv2d:
https://pythonguides.com/pytorch-mnist/

Done:
- Classification:
- - MNIST with single layer and weight visualization
- - MNIST with two layers. (Actually, not done) TODO: use softmax at the end
- - CNN MNIST. Visualize filters?
- Autoencoders:
- - AE FF MNIST.
- - AE CNN MNIST.

In all the above, add test set evaluation, possibly as a validation set.

At some point, FF and CNN might need to get their own submodules,
with their own encoder and decoder submodules? Tried it, seems messy...

For the AE models, compute binary accuracy for each batch during training
and evaluation / testing.
https://pytorch.org/torcheval/main/generated/torcheval.metrics.BinaryAccuracy.html

Also for the AE models, show how better results are with CNN, with the same
bottleneck as FF (50) and with smaller model size (1MB compared to 4MB).
See inside the code of models/cnn_models.py for details.

TODO:
- VAE CNN MNIST. https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb Only change the encoder class, to produce two outputs (mu and sigma), the main AE class for sampling in the latent and the loss function in the running script. Decoder can stay the same.
- VQVAE CNN MNIST. https://github.com/praeclarumjj3/VQ-VAE-on-MNIST
- GANs CNN MNIST. https://github.com/Ksuryateja/DCGAN-MNIST-pytorch/blob/master/gan_mnist.py