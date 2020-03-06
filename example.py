from midoria import *


# layers = (
#     Conv(knum=8, ksize=5, kchannel=1), 
#     Relu(), 
#     MaxPooling(2), 
    
#     Conv(knum=16, ksize=5, kchannel=8), 
#     Relu(), 
#     MaxPooling(2), 
    
#     Flatten(), 
    
#     Dense((64, 256)), 
#     BatchNorm((64, 1)), 
#     Relu(), 

#     Dense((10, 64)), 
#     BatchNorm((10, 1)), 
#     Sigmoid(), 
    
#     SoftmaxCrossEntropy()
# )
layers=(
    BatchNorm((784, 1)), 
    
    Dense((400, 784)), 
    BatchNorm((400, 1)), 
    Relu(), 
    
    Dense((100, 400)), 
    BatchNorm((100, 1)), 
    Relu(), 
    
    Dense((10, 100)), 
    BatchNorm((10, 1)), 
    Sigmoid(), 
    
    # MSE()
    SoftmaxCrossEntropy()
    )

optimizer = BGD(lr=0.1)

mnist = Mnist(flatten=True)

model = Model(layers, optimizer)

model.fit(
    mnist.training_images, 
    mnist.training_labels, 
    epochs=1, 
    batch_size=128)

# accuracy = model.evaluate(
#     mnist.training_images[:10000], 
#     mnist.training_labels[:, :10000])

# accuracy = model.evaluate(
#     mnist.testing_images, 
#     mnist.testing_labels)