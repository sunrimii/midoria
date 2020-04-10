# 佈署手寫辨識於網頁
![網頁示範](https://github.com/sunrimii/midoria/blob/master/demo.gif)

# 程式說明
參考[cnn_example.py](https://github.com/sunrimii/midoria/blob/master/cnn_example.py)
```python
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

# 儲存/讀取參數
model.save('cnn')
model.load('cnn')

model.evaluate(mnist.testing_images, mnist.testing_labels)
```

# 可使用層
- 一般層
  - Dense
  - Conv
- 激勵層
  - Sigmoid
  - Relu
  - Softmax
- 損失層
  - MSE
  - SoftmaxCrossEntropy
- 其他層
  - MaxPooling
  - BatchNorm
  - Flatten
