from core import *
from database.mnist import *


cnn_layers = (
    Conv(knum=8, ksize=5, kchannel=1), 
    Relu(), 
    MaxPooling(2), 
    
    Conv(knum=16, ksize=5, kchannel=8), 
    Relu(), 
    MaxPooling(2), 
    
    Flatten(), 
    
    Dense((64, 256)), 
    BatchNorm((64, 1)), 
    Relu(), 

    Dense((10, 64)), 
    BatchNorm((10, 1)), 
    Sigmoid(), 
    
    SoftmaxCrossEntropy()
)

optimizer = Adagrad(lr=0.01)
mnist = Mnist(dim=4)
model = Model(cnn_layers, optimizer)
model.fit(
    mnist.training_images, 
    mnist.training_labels, 
    epochs=1, 
    batch_size=128)

# model.save('cnn')
# model.load('cnn')

model.evaluate(mnist.testing_images, mnist.testing_labels)