from core import *
from database.mnist import *


dnn_layers=(
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

optimizer = Adagrad(lr=0.01)
mnist = Mnist(dim=2)
model = Model(dnn_layers, optimizer)
model.fit(
    mnist.training_images, 
    mnist.training_labels, 
    epochs=1, 
    batch_size=128)

# model.save('dnn')
# model.load('dnn')

model.evaluate(mnist.testing_images, mnist.testing_labels)

# # 預測單張
# index = np.random.randint(0, 1000, 1)
# image = mnist.testing_images[:, index]
# label = mnist.testing_labels[:, index]
# print(f'answer{label.argmax()}')
# model.predict(image)