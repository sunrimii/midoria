from midoria import *


layers=(
    Dense((32, 784)), 
    Sigmoid(), 
    Dense((16, 32)), 
    Sigmoid(), 
    Dense((10, 16)), 
    Sigmoid(), 
    MSE())

optimizer = BGD(learning_rate=1, batch_size=100)

mnist = Mnist()

network = Network(mnist, layers, optimizer)
network.train(epoch=1)
network.accuracy()