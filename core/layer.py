import numpy as np


class Layer:
    def __init__(self):
        self.previous_layer: Layer = None
        self.next_layer: Layer = None

class BatchNorm(Layer):
    def __init__(self, shape):
        super().__init__()
        
        self.gamma = np.random.uniform(0.9, 1.1, shape)
        self.beta = np.random.uniform(-0.1, 0.1, shape)

        # 根據模式的不同使用不同組參數
        self.do_fit = False

    def forward(self, x):
        # 在訓練階段使用基於批量計算的參數
        if self.do_fit:
            # 平均值
            self.u = x.mean(axis=1, keepdims=True)
            # 方差
            v = x.var(axis=1, keepdims=True)
            # 標準差
            self.s = (v + 1e-15) ** 0.5
        # 在評估階段使用基於整體計算的參數
        else:
            if not hasattr(self, 'overall_u'):
                # 平均值
                self.overall_u = x.mean(axis=1, keepdims=True)
                # 方差
                v = x.var(axis=1, keepdims=True)
                # 標準差
                self.overall_s = (v + 1e-15) ** 0.5

            self.u = self.overall_u
            self.s = self.overall_s

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

class Flatten(Layer):
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

class Conv(Layer):
    def __init__(self, knum, ksize, kchannel, strides=1, pwidth=0):
        super().__init__()
        
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
        
class MaxPooling(Layer):
    def __init__(self, ksize):
        super().__init__()
        
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

class Dense(Layer):
    def __init__(self, shape):
        super().__init__()
        
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

class Sigmoid(Layer):
    def forward(self, z):
        # self.a = 1 / (1 + np.exp(-z))
        self.a = 0.5 * (1 + np.tanh(0.5 * z))
        self.next_layer.forward(self.a)
    
    def backward(self, delta):
        dadz = self.a * (1 - self.a)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class Relu(Layer):
    def forward(self, z):
        self.z = z
        a = np.maximum(z, 0)
        self.next_layer.forward(a)
    
    def backward(self, delta):
        dadz = np.where(self.z > 0, 1, 0)
        dLdz = delta * dadz
        self.previous_layer.backward(dLdz)

class Softmax(Layer):
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

class MSE(Layer):
    def forward(self, a):
        self.prediction = self.a = a
    
    def backward(self, y):
        dLda = 2 * (y - self.a)
        self.previous_layer.backward(dLda)

    def calc_loss(self, y):
        loss = ((y - self.a) ** 2).sum(axis=0).mean()
        return loss

class SoftmaxCrossEntropy(Layer):
    def forward(self, z):
        # 避免溢出
        z -= z.max(axis=0)

        expz = np.exp(z)
        self.prediction = self.a = expz / expz.sum(axis=0)

    def backward(self, y):
        dLdz = self.a - y
        self.previous_layer.backward(dLdz)

    def calc_loss(self, y):
        loss = -(y * np.log10(self.a)).sum(axis=0).mean()
        return loss