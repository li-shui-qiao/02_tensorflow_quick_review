## Convolutional Network

通过使用卷积层和池化层处理图片数据，之后导入训练模型。  
模型依旧使用顺序模型：

* 导入库函数：
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
```

* 首先定义一个顺序模型再导入卷积层：
```
model.add(
    Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),data_format='channels_last')
)
```
  * 第一个参数为units，第二个为每次选取的kernel_size。详细参考：[Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)

* 接下来导入池化层：
```
'''
three by three filter
'''
model.add(
    MaxPooling2D((3,3))
)
```
  * (3,3)为池化范围因为我们卷机kernel_size选择了(3,3)。MaxPooling为在选定子范围  
  提取出最大值，详细参考[MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
  
* 导入optimizer并设置学习率：
```
'''
define optimizer, set learning rate to 0.005
'''
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
```
  * 学习率设置为0.005，较小就需要更长的epoch较大变化较快，需要较小epoch。
  详细参考：[Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer)
  
* 设置准确度和误差函数：
```
'''
set accurancy， 设置准确度
'''
acc = tf.keras.metrics.SparseCategoricalAccuracy()
 
'''
set error，设置误差函数
'''
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt, loss='sparse_categorical_crossentropy', metrics=[acc,mae])
```
  * 设置准确度和误差函数不需要参数，设置compile具体参数参考：[Compile](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile)
  
* 导入matplotlib和numpy：
```
import matplotlib.pyplot as plt
import numpy as np
```
  * matplotlib被用来python数据绘制图表和进行图表编辑。详情参考：[matplotlib](https://matplotlib.org/)
  * numpy被用来处理复杂的多维数组。详情参考：[numpy](https://numpy.org/)
  
* 导入tensorflow官方的图片库用作例子：
```
'''
import official example dataset

https://www.tensorflow.org/tutorials/keras/classification
'''
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist_data.load_data()
```
  * fashion_mnist库是tensorflow较为出名的图库，里面存放多种服装的正面图
  
* 设置标签：
```
'''
class names
'''
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
  * 图库标签分类
  
* fit模型
```
'''
input data, epochs循环次数， batch——size一次性写入数据量，注意写入数据有一组颜色纬度，需要额外为train——images创建一维
'''
history = model.fit(train_images[:,:,:,np.newaxis], train_labels, epochs=2, batch_size=256)
```
  * epochs为循环次数， batch_size为一次性写入量。 注意写入的图片有一组颜色为度，需要额外为train_images额外创建一维
  详情参考：[Fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
  
## 检测结果

* 选取一张随机图片：
```
random_inx = np.random.choice(test_images.shape[0])
test = test_images[random_inx]
plt.imshow(test)
plt.show()
print("label: ",class_names[test_labels[random_inx]])
```

* 检测结果：
```
prediction = model.predict(test[np.newaxis,:,:,np.newaxis])
print("prediction: ", class_names[np.argmax(prediction)])
```
  * predict用法：[predict](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)
  * 检测结果与随机图片一致：
  ```
  label:  Trouser
  prediction:  Trouser
  ```
  







