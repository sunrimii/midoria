import base64
import cv2
from flask import Flask, render_template, request
import numpy as np

from core import *


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # uri轉numpy
        uri = request.values['uri']
        string = uri.split(',', 1)[1].replace(' ', '+')
        byteimage = base64.b64decode(string)
        image = np.frombuffer(byteimage, dtype=np.int8)
        
        # 縮小
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))

        # 黑白反轉
        image = 255 - image
        
        # 儲存最近的輸入
        cv2.imwrite('userinput.jpg',image)

        # 預測
        image = image.reshape(1, 28, 28, 1) / 255
        prediction = model.predict(image)
        return str(prediction)


# 初始化模型且需要傳入優化器
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
model = Model(cnn_layers, optimizer)

# 讀取已訓練的參數
model.load('cnn913')

app.run(host='192.168.1.100', port='5278')
