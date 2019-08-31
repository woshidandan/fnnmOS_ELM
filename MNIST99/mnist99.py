import pandas as pd
import numpy as np

from model import OSELM
import numpy as np

import numpy as np
import tensorflow as tf
import random as rn
from keras.engine import Model

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import seaborn as sns
# %matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)

# free some space
# del train
#
# g = sns.countplot(Y_train)
#
# Y_train.value_counts()
#
# X_train.isnull().any().describe()
#
# test.isnull().any().describe()

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 2

X_train,X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
# history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
#          validation_data = (X_val, Y_val), verbose = 2)
model.summary()
model.load_weights("mnist99.h5")
# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
#
#
# datagen.fit(X_train)
#
# history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
#                               epochs = epochs, validation_data = (X_val,Y_val),
#                               verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
#                               , callbacks=[learning_rate_reduction])

def hidden_layer_generate(model):

    """
    CNNの中間層の出力を取得するモデルの構築
    :param cnn_model: CNNモデル
    :return:
    """

    layer_name = 'dense_2'
    hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    cnn_train_result = hidden_layer_model.predict(X_train)
    cnn_test_result = hidden_layer_model.predict(X_val)
    return hidden_layer_model, cnn_train_result, cnn_test_result

tf.set_random_seed(2016)  #随机序列可重复
sess = tf.Session()

path = "./OSELM.h5"
hidden_num = 7
###
batch_size = 1000
###
# # load OS-ELM model
# elm = OSELM(sess, batch_size, 10, hidden_num, 10)
# hidden_layer_model, cnn_train_result, cnn_test_result= hidden_layer_generate(model)
# elm.load(sess, 'OSELM.h5')
# elm.test(cnn_test_result, Y_val)

'''
将循环去除后，可正常运行
考虑可能是sess的原因
'''

elm = OSELM(sess, batch_size, 10, hidden_num, 10)
#
# data_train_2d, data_test_2d, target_train, target_test = load_mnist_2d()
# # print(target_train.shape)  1203
# cnn_model = cnn_generate(data_train_2d, target_train)

hidden_layer_model, cnn_train_result, cnn_test_result = hidden_layer_generate(model)


epoch = 0
data_container = []
target_container = []
target_train_convert = Y_train


while(epoch < 20):
        k = 0

        for (index,data) in enumerate(cnn_train_result, start= 0 ):
            if(index >= (epoch) * batch_size and index < (epoch+1) * batch_size):
            # if (index >= (epoch ) * batch_size and index < (epoch +1) * batch_size):
                data_container.append(data)
                k += 1
                if k == batch_size:
                    break

        j = 0
        for (index1,target) in enumerate(target_train_convert, start= 0):
            if (index1 >= (epoch) * batch_size and index1 < (epoch + 1) * batch_size):
            # if (index >= (epoch) * batch_size and index < (epoch + 1) * batch_size):
                target_container.append(target)
                j += 1
                if j == batch_size:
                    break
        data_container = np.array(data_container)
        target_container = np.array(target_container)
        elm.train(data_container, target_container)
        # 保存ELM训练得到的结果
        elm.save(sess, path)
        # elm.test(X_test, Y_test)
        # 输出精度
        elm.test(cnn_test_result, Y_val)
        # elm.test(data_container, target_container)


        # print(epoch)
        epoch += 1
        data_container = []
        target_container = []
# elm.save(sess, './OSELM.h5')
# train OS-ELM model
# while(hidden_num < 11):
#
#     elm = OSELM(sess, batch_size, 10, hidden_num, 10)
#     #
#     # data_train_2d, data_test_2d, target_train, target_test = load_mnist_2d()
#     # # print(target_train.shape)  1203
#     # cnn_model = cnn_generate(data_train_2d, target_train)
#
#     hidden_layer_model, cnn_train_result, cnn_test_result= hidden_layer_generate(model)
#     # print("cnn_train_result")  #(42500, 10)
#     # print(cnn_train_result.shape)
#
#     epoch = 0
#     data_container = []
#     target_container = []
#     target_train_convert = Y_train
#
#     while(epoch < 20):
#         k = 0
#
#         for (index,data) in enumerate(cnn_train_result, start= 0 ):
#             if(index >= (epoch) * batch_size and index < (epoch+1) * batch_size):
#             # if (index >= (epoch ) * batch_size and index < (epoch +1) * batch_size):
#                 data_container.append(data)
#                 k += 1
#                 if k == batch_size:
#                     break
#
#         j = 0
#         for (index1,target) in enumerate(target_train_convert, start= 0):
#             if (index1 >= (epoch) * batch_size and index1 < (epoch + 1) * batch_size):
#             # if (index >= (epoch) * batch_size and index < (epoch + 1) * batch_size):
#                 target_container.append(target)
#                 j += 1
#                 if j == batch_size:
#                     break
#         data_container = np.array(data_container)
#         target_container = np.array(target_container)
#         elm.train(data_container, target_container)
#         elm.save(sess, path)
#         # elm.test(X_test, Y_test)
#         elm.test(cnn_test_result, Y_val)
#         # elm.test(data_container, target_container)
#
#
#         # print(epoch)
#         epoch += 1
#         data_container = []
#         target_container = []
#
#     print(hidden_num)
#     hidden_num = 1 + hidden_num
#

# Save OS-ELM model
# elm.save(sess, './OSELM.h5')