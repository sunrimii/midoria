import numpy as np


ONE_HOT = np.identity(10)


class Flatten:
    pass

class Conv:
    def __init__(self, kshpae, strides=1, pad_method='valid'):
        self.W = 2 * np.random.rand(*kshpae) - 1
        self.b = np.zeros(kshpae[:2])

        # 優化器用
        self.W_batch_gradient = 0
        self.b_batch_gradient = 0

        self.ksize = ksize
        self.strides = strides
        self.pad_method = pad_method

    def pad(self, x, method):
        # 不填充
        if method == 'valid':
            pass
        # 與原來相同
        elif method == 'same':
            p = self.ksize // 2
        # 比原來還大
        elif method == 'full':
            p = self.ksize - 1

        x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        return x
    
    def split(self, x):
        xnum, xsize, xsize, xchannel = x.shape
        self.zsize = (xsize - self.ksize) // self.strides + 1
        shape = (xnum, self.zsize, self.zsize, self.ksize, self.ksize, xchannel)
        strides = (x.strides[0], x.strides[1]*self.strides, x.strides[2]*self.strides, *x.strides[1:])
        x = np.lib.stride_tricks.as_strided(x, shape, strides)
        return x

    def forward(self, x):
        x = self.pad(x, self.pad_method)
        x = self.split(x)
        z = np.tensordot(self.W, x) + self.b
        self.next_layer.forward(z)
    
    def backward(self, delta):
        # 缺少dLdW
        

        
        if self.previous_layer is not None:
            if self.strides > 1:
                zero = np.zeros((self.zsize, self.zsize))
                zero[::self.strides, ::self.strides] = delta
                delta = zero
            delta = self.pad(delta, 'full')
            delta = self.split(delta)
            dadx = np.rot90(self.W, 2)
            dLdx = np.tensordot(delta, dadx)
            self.previous_layer.backward(dLdx)
        
class MaxPooling:
    def __init__(self, ksize):
        self.ksize = ksize

    def split(self, a):
        anum, asize, asize, achannel = a.shape
        self.zsize = asize // self.ksize
        shape = (anum, self.zsize, self.zsize, self.ksize, self.ksize, achannel)
        strides = (a.strides[0], a.strides[1]*2, a.strides[2]*2, *a.strides[1:])
        a = np.lib.stride_tricks.as_strided(a, shape, strides)
        return a
    
    def forward(self, a):
        a = self.split(a)
        a = a.reshape(a.shape[0], -1, a.shape[3])
        a = a.max(axis=1)
        z = a.reshape(a.shape[0], a.shape[1]//self.ksize, a.shape[2]//self.ksize, a.shape[3])

        # 反向傳播用
        self.mask = z.repeat(self.ksize, axis=1).repeat(self.ksize, axis=2) != a
        
        self.neat_layer.forward(z)

    def backward(self, delta):
        delta = delta.repeat(self.ksize, axis=1).repeat(self.ksize, axis=2)
        dLdx = delta * self.mask
        self.previous_layer.backward(dLdx)

class Dense:
    def __init__(self, shape):
        self.W = 2 * np.random.rand(*shape) - 1
        self.b = np.random.rand(shape[0], 1)

        # 優化器用
        self.W_batch_gradient = 0
        self.b_batch_gradient = 0
        
    def forward(self, x):
        # 反向傳播用
        self.x = x

        z = np.dot(self.W, x) + self.b
        self.next_layer.forward(z)
    
    def backward(self, delta):
        # 優化器用
        self.W_batch_gradient += np.dot(delta, self.x.T)
        self.b_batch_gradient += delta
        
        if self.previous_layer is not None:
            dzdx = self.W.T
            dLdx = np.dot(dzdx, delta)
            self.previous_layer.backward(dLdx)

class Sigmoid:
    def forward(self, z):
        # 反向傳播用
        self.z = z

        a = 1 / (1 + np.exp(-z))
        self.next_layer.forward(a)
    
    def backward(self, delta):
        dadz = np.exp(-self.z) / ((1 + np.exp(-self.z)) ** 2)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class Relu:
    def forward(self, z):
        # 反向傳播用
        self.z = z

        a = np.maximum(z, 0)
        self.next_layer.forward(a)
    
    def backward(self, delta):
        dadz = np.where(self.z > 0, 1, 0)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class Softmax:
    def forward(self, z):
        # 避免溢出
        z -= max(z)
        
        # 反向傳播用
        self.a = np.exp(z) / sum(np.exp(z))

        self.next_layer.forward(self.a)
    
    def backward(self, delta):
        dadz = self.a * (1 - self.a)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class MSE:
    def forward(self, a):
        # 反向傳播用 預測用
        self.a = a

    def backward(self, y):
        dLda = 2 * (y - self.a)
        self.previous_layer.backward(dLda)

class SoftmaxCrossEntropy:
    def forward(self, z):
        # 避免溢出
        z -= max(z)

        # 反向傳播用 預測用
        self.a = np.exp(z) / sum(np.exp(z))

    def backward(self, y):
        dLdz = self.a - y
        self.previous_layer.backward(dLdz)

class BGD:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
    
        self.batch_size = batch_size
        self.batch_count = 1

    def update(self):
        self.batch_count += 1
        if self.batch_count == self.batch_size:
            for layer in self.Dense_layers:
                layer.W += self.learning_rate * layer.W_batch_gradient / self.batch_size
                layer.b += self.learning_rate * layer.b_batch_gradient / self.batch_size
        
                layer.W_batch_gradient = 0
                layer.b_batch_gradient = 0
            
            self.batch_count = 1

class Network:
    def __init__(self, data, layers, optimizer):
        self.data = data
        self.layers = layers
        self.optimizer = optimizer

        # 參數只在線性層
        self.optimizer.Dense_layers = [layer for layer in self.layers if isinstance(layer, Dense)]

        # 建立前後層關係
        for i in range(len(layers) - 1):
            self.layers[i].next_layer = self.layers[i+1]
        for i in range(1, len(layers)):
            self.layers[i].previous_layer = self.layers[i-1]
        self.layers[0].previous_layer = None
        self.layers[-1].next_layer = None

        self.input_layer = self.layers[0]
        self.loss_layer = self.layers[-1]

    def train(self, epoch):
        for _ in range(epoch):
            # 打亂訓練集
            order = np.arange(len(self.data.training_images))
            np.random.shuffle(order)
            
            for i in order:
                x = self.data.training_images[i]
                y = self.data.training_labels[i]

                x = x.reshape(-1,1)
                y = ONE_HOT[y].reshape(-1, 1)

                self.input_layer.forward(x)
                self.loss_layer.backward(y)
            
                self.optimizer.update()

    def predict(self, x):
        self.input_layer.forward(x)
        prediction = self.loss_layer.a.argmax()
        return prediction
        
    def accuracy(self):
        # 訓練集的20%作為驗證集
        correct = 0
        data_len = len(self.data.training_images) // 5
        for i in range(data_len):
            x = self.data.training_images[i].reshape(-1, 1)
            y = self.data.training_labels[i]

            if self.predict(x) == y:
                correct += 1
        rate = correct / data_len * 100
        print(f"驗證集準確率: {rate:.2f}%")
        
        correct = 0
        data_len = len(self.data.testing_images)
        for i in range(data_len):
            x = self.data.testing_images[i].reshape(-1, 1)
            y = self.data.testing_labels[i]

            if self.predict(x) == y:
                correct += 1
        rate = correct / data_len * 100
        print(f"測試集準確率: {rate:.2f}%")

class Mnist:
    def __init__(self):
        self.training_images = self.load_image('database/mnist/train-images.idx3-ubyte')
        self.training_labels = self.load_label('database/mnist/train-labels.idx1-ubyte')
        self.testing_images = self.load_image('database/mnist/t10k-images.idx3-ubyte')
        self.testing_labels = self.load_label('database/mnist/t10k-labels.idx1-ubyte')

    def load_image(self, path):
        with open(path, 'rb') as file:
            file.read(16)
            return np.fromfile(file, dtype=np.uint8).reshape(-1, 784) / 255
        
    def load_label(self, path):
        with open(path, 'rb') as file:
            file.read(8)
            return np.fromfile(file, dtype=np.uint8)