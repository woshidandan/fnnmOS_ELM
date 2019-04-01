from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Reshape,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv1D,MaxPooling1D
import numpy as np
import os
import keras

import tensorflow as tf
from model import OSELM
import numpy as np
from keras.engine import Model
model_name = 'boston.h5'
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_test.shape)
# y_train = keras.utils.to_categorical(y_train, 1)
print(y_test.shape)
model = Sequential(name='boston')
model.add(BatchNormalization(input_shape=(13,)))
model.add(Reshape((13, 1,1)))
model.add(Conv2D(filters=13, strides=1, padding='same', kernel_size=1, activation='sigmoid'))
model.add(Conv2D(filters=26, strides=2, padding='same', kernel_size=2, activation='sigmoid'))
model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
model.add(Conv2D(filters=52, strides=1, padding='same', kernel_size=1, activation='sigmoid'))
model.add(Conv2D(filters=104, strides=2, padding='same', kernel_size=2, activation='sigmoid'))
model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1))
model.load_weights(model_name)

model.compile('adam','mae',metrics=['accuracy'])
model.summary()
# history=model.fit(x_train,y_train,batch_size=404,epochs=10000,verbose=1,validation_data=(x_test,y_test))
# model.save(model_name)
# print(history.history)
# print("acc:{},loss:{}".format(history.history['acc'],history.history['loss']))
# f=open("result.txt",'a')
# f.write(str(history.history['val_loss'][-1])+"\n")
# f.close()

def hidden_layer_generate(model):

    """
    CNNの中間層の出力を取得するモデルの構築
    :param cnn_model: CNNモデル
    :return:
    """

    layer_name = 'flatten_1'
    hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    cnn_train_result = hidden_layer_model.predict(x_train)
    cnn_test_result = hidden_layer_model.predict(x_test)
    return hidden_layer_model, cnn_train_result, cnn_test_result



hidden_num = 1
###
batch_size = 5
###


tf.set_random_seed(2016)  #随机序列可重复
sess = tf.Session()
# while(hidden_num < 700):
elm = OSELM(sess, batch_size, 416, hidden_num, 1)
#
# data_train_2d, data_test_2d, target_train, target_test = load_mnist_2d()
# # print(target_train.shape)  1203
# cnn_model = cnn_generate(data_train_2d, target_train)

hidden_layer_model, cnn_train_result, cnn_test_result= hidden_layer_generate(model)
# print(cnn_train_result.shape)
# print("cnn_train_result")  #(42500, 10)
# print(cnn_train_result.shape)



epoch = 0
data_container = []
target_container = []
# target_train_convert = Y_train
# print("Y_train")   (42500, 10)
# print(Y_train.shape)
# target_test_convert = np_utils.to_categorical(target_test, NUM_CLASS)

#test
# print("X_test")
# print(X_test.shape)
# Y_test = np.array(Y_test)
# print("Y_test")
# print(Y_test.shape)

# print("hidden_num：{},batch_size: {}".format(hidden_num,batch_size))
# print("-------------------------------------------------------------------")
while(epoch < 40):
    k = 0
    y_test_use = y_test

    for (index,data) in enumerate(cnn_train_result, start= 0 ):
        if(index >= (epoch) * batch_size and index < (epoch+1) * batch_size):
        # if (index >= (epoch ) * batch_size and index < (epoch +1) * batch_size):
        #     print(data.shape)
            data_container.append(data)
            k += 1
            if k == batch_size:
                break
    # print(data_container.shape)

    j = 0
    for (index1,target) in enumerate(y_train, start= 0):
        if (index1 >= (epoch) * batch_size and index1 < (epoch + 1) * batch_size):
        # if (index >= (epoch) * batch_size and index < (epoch + 1) * batch_size):
            target_container.append(target)
            j += 1
            if j == batch_size:
                break
    # print(target_container)
    data_container = np.array(data_container)

    target_container = np.array(target_container)
    target_container = target_container[:, np.newaxis] #将一维(32,)转化为二维(32,1)


    elm.train(data_container, target_container)
    # elm.test(data_container, target_container)
    # # elm.test(X_test, Y_test)
    # y_test = y_test[:, np.newaxis]
    y_test_use = y_test_use[:, np.newaxis]
    elm.test(cnn_test_result, y_test_use)
    # # elm.test(data_container, target_container)


    # print(epoch)
    epoch += 1
    data_container = []
    target_container = []
# hidden_num = 10 + hidden_num;
# print(hidden_num)