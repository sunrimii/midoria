import pickle

import numpy as np

from .layer import Dense, Conv, BatchNorm


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

        # 取輸入層損失層的別名
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def fit(self, training_x, training_y, epochs, batch_size):
        # 切換歸一化層成訓練狀態
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.do_fit = True

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
                self.output_layer.backward(batch_y)
                self.optimizer.update()

                # 每隔幾個批量顯示準確率
                if i % 50 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, ", end='')
                    self.evaluate(batch_x, batch_y)

        # 使歸一化層準備在評估時所需要的參數
        # 以非訓練模式前向傳播時
        # 若該層沒有基於整體計算的參數才會準備
        # 因此不會每次訓練都重複準備
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.do_fit = False
        self.input_layer.forward(training_x)

        # 顯示模型參數
        print(f"Total epochs: {epochs}, Batch size: {batch_size}, Learning rate: {self.optimizer.lr}")

    def predict(self, x):
        self.input_layer.forward(x)
        prediction = self.output_layer.prediction.argmax()
        print(f"prediction: {prediction}")
        return prediction

    def evaluate(self, batch_x, batch_y):
        self.input_layer.forward(batch_x)
        accuracy = (self.output_layer.prediction.argmax(axis=0) == batch_y.argmax(axis=0)).mean() * 100
        loss = self.output_layer.calc_loss(batch_y)
        print(f"accuracy: {accuracy:.1f}, loss: {loss:.3f}")
        return accuracy, loss

    def save(self, name):
        params = {}

        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                params[f'BatchNorm{i}gamma'] = layer.gamma
                params[f'BatchNorm{i}beta'] = layer.beta
                params[f'BatchNorm{i}overall_u'] = layer.overall_u
                params[f'BatchNorm{i}overall_s'] = layer.overall_s
            elif isinstance(layer, Dense):
                params[f'Dense{i}W'] = layer.W
                params[f'Dense{i}b'] = layer.b
            elif isinstance(layer, Conv):
                params[f'Conv{i}W'] = layer.W
                params[f'Conv{i}b'] = layer.b

        np.savez(f'param/{name}', **params)
        print(f"Parameters have been saved in {name}.npz.")

    def load(self, name):
        params = np.load(f'param/{name}.npz')
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                layer.gamma = params[f'BatchNorm{i}gamma']
                layer.beta = params[f'BatchNorm{i}beta']
                layer.overall_u = params[f'BatchNorm{i}overall_u']
                layer.overall_s = params[f'BatchNorm{i}overall_s']
            elif isinstance(layer, Dense):
                layer.W = params[f'Dense{i}W']
                layer.b = params[f'Dense{i}b']
            elif isinstance(layer, Conv):
                layer.W = params[f'Conv{i}W']
                layer.b = params[f'Conv{i}b']
                      