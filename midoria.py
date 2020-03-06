import numpy as np


class BatchNorm:
    def __init__(self, shape):
        self.gamma = np.random.uniform(0.99, 1.01, shape)
        self.beta = np.random.uniform(-0.01, 0.01, shape)

        self.do_predict_or_evaluate = False

    def forward(self, x):
        # 在訓練階段使用基於批量計算的參數
        # 在評估階段使用基於整體計算的參數
        if not self.do_predict_or_evaluate:
            # 平均值
            self.u = x.mean(axis=1, keepdims=True)
            # 方差
            v = x.var(axis=1, keepdims=True)
            # 標準差
            self.s = (v + 1e-15) ** 0.5

        self.xhat = (x - self.u) / self.s
        z = self.gamma * self.xhat + self.beta
        self.next_layer.forward(z)
        
    def backward(self, delta):
        # 優化器用
        self.dLdgamma = (delta * self.xhat).mean(axis=1, keepdims=True)
        self.dLdbeta = delta.mean(axis=1, keepdims=True)
        
        if self.previous_layer is not None:
            dLdx = self.gamma / self.s * (delta - self.dLdbeta - self.dLdgamma * self.xhat)
            self.previous_layer.backward(dLdx)

class Flatten:
    def forward(self, x):
        self.xshape = x.shape

        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        x = np.rot90(x)
        self.next_layer.forward(x)

    def backward(self, delta):
        delta = np.rot90(delta, -1)
        delta = delta.reshape(self.xshape)
        self.previous_layer.backward(delta)

class Conv:
    def __init__(self, knum, ksize, kchannel, strides=1, pwidth=0):
        self.W = np.random.uniform(-0.01, 0.01, (knum, ksize, ksize, kchannel))
        self.b = np.random.uniform(-0.01, 0.01, (knum, ksize, ksize, kchannel))

        self.ksize = ksize
        self.strides = strides
        self.pwidth = pwidth

    def split_by_strides(self, x):
        xnum, xsize, xsize, xchannel = x.shape
        self.zsize = (xsize - self.ksize) // self.strides + 1
        shape = (xnum, self.zsize, self.zsize, self.ksize, self.ksize, xchannel)
        strides = (x.strides[0], x.strides[1]*self.strides, x.strides[2]*self.strides, *x.strides[1:])
        splited_x = np.lib.stride_tricks.as_strided(x, shape, strides)
        return splited_x

    def forward(self, x):
        padded_x = np.pad(x, ((0, 0), (self.pwidth, self.pwidth), (self.pwidth, self.pwidth), (0, 0)), 'constant')

        self.splited_x = self.split_by_strides(padded_x)
        self.splited_1 = np.ones_like(self.splited_x)

        # z = x conv w
        # z = np.tensordot(self.splited_x, self.W, axes=((3, 4, 5), (1, 2, 3)))
        z = np.tensordot(self.splited_x, self.W, axes=((3, 4, 5), (1, 2, 3))) + \
            np.tensordot(self.splited_1, self.b, axes=((3, 4, 5), (1, 2, 3)))
        self.next_layer.forward(z)
    
    def backward(self, delta):
        # 優化器用
        # dLdw = x conv delta
        # dLdb = 1 conv delta
        batch_size = delta.shape[0]
        self.dLdW = np.tensordot(delta, self.splited_x, axes=((0, 1, 2), (0, 1, 2))) / batch_size
        self.dLdb = np.tensordot(delta, self.splited_1, axes=((0, 1, 2), (0, 1, 2))) / batch_size

        if self.previous_layer is not None:
            if self.strides > 1:
                zero = np.zeros((self.zsize, self.zsize))
                zero[::self.strides, ::self.strides] = delta
                delta = zero
            
            # dLdx = pad(delta) conv rot180(W)
            pwidth = self.ksize - 1
            padded_delta = np.pad(delta, ((0, 0), (pwidth, pwidth), (pwidth, pwidth), (0, 0)), 'constant')
            splited_delta = self.split_by_strides(padded_delta)
            rot180_W = np.rot90(self.W, 2)
            dLdx = np.tensordot(splited_delta, rot180_W.T, axes=((3, 4, 5), (1, 2, 3)))
            self.previous_layer.backward(dLdx)
        
class MaxPooling:
    def __init__(self, ksize):
        self.ksize = ksize
    
    def forward(self, a):
        anum, asize, asize, achannel = a.shape
        zsize = asize // self.ksize
        z = a.reshape(anum, zsize, self.ksize, zsize, self.ksize, achannel).max(axis=(2, 4))
        
        self.mask = z.repeat(self.ksize, axis=1).repeat(self.ksize, axis=2) != a
        
        self.next_layer.forward(z)

    def backward(self, delta):
        dLdx = delta.repeat(self.ksize, axis=1).repeat(self.ksize, axis=2) * self.mask
        self.previous_layer.backward(dLdx)

class Dense:
    def __init__(self, shape):
        self.W = np.random.uniform(-0.01, 0.01, shape)
        self.b = np.random.uniform(-0.01, 0.01, (shape[0], 1))
        
    def forward(self, x):
        self.x = x

        z = np.dot(self.W, x) + self.b
        self.next_layer.forward(z)
    
    def backward(self, delta):
        # 優化器用
        batch_size = delta.shape[1]
        self.dLdW = np.dot(delta, self.x.T) / batch_size
        self.dLdb = delta.mean(axis=1, keepdims=True)
        
        if self.previous_layer is not None:
            dzdx = self.W.T
            dLdx = np.dot(dzdx, delta)
            self.previous_layer.backward(dLdx)

class Sigmoid:
    def forward(self, z):
        self.a = 1 / (1 + np.exp(-z))
        self.next_layer.forward(self.a)
    
    def backward(self, delta):
        dadz = self.a * (1 - self.a)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class Relu:
    def forward(self, z):
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
        z -= z.max(axis=0, keepdims=True)
        
        expz = np.exp(z)
        self.a = expz / expz.sum(axis=0)
        self.next_layer.forward(self.a)
    
    def backward(self, delta):
        dadz = self.a * (1 - self.a)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class MSE:
    def forward(self, a):
        self.a = a

        # 取別名
        self.prediction = self.a
    
    def backward(self, y):
        dLda = 2 * (y - self.a)
        self.previous_layer.backward(dLda)

class SoftmaxCrossEntropy:
    def forward(self, z):
        # 避免溢出
        z -= z.max(axis=0)

        expz = np.exp(z)
        self.a = expz / expz.sum(axis=0)
        
        # 取別名
        self.prediction = self.a

    def backward(self, y):
        dLdz = self.a - y
        self.previous_layer.backward(dLdz)

class BGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.gamma += self.lr * layer.dLdgamma
                layer.beta += self.lr * layer.dLdbeta
            elif isinstance(layer, (Dense, Conv)):
                layer.W += self.lr * layer.dLdW
                layer.b += self.lr * layer.dLdb

class Model:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer
        self.optimizer.layers = self.layers

        # 建立前後層關係
        for i in range(len(layers) - 1):
            self.layers[i].next_layer = self.layers[i+1]
        for i in range(1, len(layers)):
            self.layers[i].previous_layer = self.layers[i-1]
        self.layers[0].previous_layer = None
        self.layers[-1].next_layer = None

        # 取輸入層損失層的別名
        self.input_layer = self.layers[0]
        self.loss_layer = self.layers[-1]

    def fit(self, training_x, training_y, epochs, batch_size):
        # 存放訓練過程的詳細資訊
        # history = {}
        
        order = np.arange(training_y.shape[1])
        for epoch in range(epochs):
            # 打亂順序
            np.random.shuffle(order)

            # 分配成批量
            batches = []
            for i in range(0, len(order), batch_size):
                if training_x.ndim == 2:
                    try:
                        batches.append((
                            training_x[:, order[i: i+batch_size]], 
                            training_y[:, order[i: i+batch_size]]))
                    except IndexError:
                        batches.append((
                            training_x[:, order[i:]], 
                            training_y[:, order[i:]]))
                elif training_x.ndim == 4:
                    try:
                        batches.append((
                            training_x[order[i: i+batch_size]], 
                            training_y[:, order[i: i+batch_size]]))
                    except IndexError:
                        batches.append((
                            training_x[order[i:]], 
                            training_y[:, order[i:]]))
            
            # 訓練每個批量 
            for i, (batch_x, batch_y) in enumerate(batches):
                self.input_layer.forward(batch_x)
                self.loss_layer.backward(batch_y)
                self.optimizer.update()

                if i % 50 == 0:
                    accuracy = self.evaluate(batch_x, batch_y)
                    print(f"Epoch: {epoch}, Batch accuracy: {accuracy:.3f}")

        # 使批量標準化層準備在評估時所需要的參數
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                self.input_layer.forward(training_x)
                break
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.do_predict_or_evaluate = True
    
    def evaluate(self, batch_x, batch_y):
        self.input_layer.forward(batch_x)
        accuracy = (self.loss_layer.prediction.argmax(axis=0) == batch_y.argmax(axis=0)).mean()
        return accuracy

class Mnist:
    def __init__(self, flatten=False):
        self.flatten = flatten
        
        self.training_images = self.load_images('database/mnist/train-images.idx3-ubyte')
        self.training_labels = self.load_labels('database/mnist/train-labels.idx1-ubyte')
        self.testing_images = self.load_images('database/mnist/t10k-images.idx3-ubyte')
        self.testing_labels = self.load_labels('database/mnist/t10k-labels.idx1-ubyte')

    def load_images(self, path):
        with open(path, 'rb') as file:
            file.read(16)
            if self.flatten:
                images = np.fromfile(file, dtype=np.uint8).reshape(-1, 28*28) / 255
                images = np.rot90(images)
                images = np.flipud(images)
            else:
                images = np.fromfile(file, dtype=np.uint8).reshape(-1, 28, 28, 1) / 255
            return images

    def load_labels(self, path):
        with open(path, 'rb') as file:
            file.read(8)
            labels = np.fromfile(file, dtype=np.uint8)
            one_hot_labels = np.zeros((10, len(labels)))
            for i, l in enumerate(labels):
                one_hot_labels[l, i] = 1
            return one_hot_labels
