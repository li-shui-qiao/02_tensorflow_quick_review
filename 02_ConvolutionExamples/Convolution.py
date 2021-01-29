#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:47:29 2021

@author: li-shui-qiao
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf

model = Sequential()

'''
1st neuron units, 2nd kernel_size
'''
model.add(
    Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),data_format='channels_last')
)

'''
three by three filter
'''
model.add(
    MaxPooling2D((3,3))
)

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

model.summary()


#优化

'''
define optimizer, set learning rate to 0.005
'''
opt = tf.keras.optimizers.Adam(learning_rate=0.005)

'''
set accurancy， 设置准确度
'''
acc = tf.keras.metrics.SparseCategoricalAccuracy()
 
'''
set error，设置误差函数
'''
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt, loss='sparse_categorical_crossentropy', metrics=[acc,mae])

#训练模型

import matplotlib.pyplot as plt
import numpy as np

'''
import official example dataset

https://www.tensorflow.org/tutorials/keras/classification
'''
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist_data.load_data()

'''
class names,  设置标签
'''
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

'''
input data, epochs循环次数， batch——size一次性写入数据量，注意写入数据有一组颜色纬度，需要额外为train——images创建一维
'''
history = model.fit(train_images[:,:,:,np.newaxis], train_labels, epochs=2, batch_size=256)

'''
选取随机图片
'''
random_inx = np.random.choice(test_images.shape[0])
test = test_images[random_inx]
plt.imshow(test)
plt.show()
print("label: ",class_names[test_labels[random_inx]])

'''
predict
'''
prediction = model.predict(test[np.newaxis,:,:,np.newaxis])
print("prediction: ", class_names[np.argmax(prediction)])










