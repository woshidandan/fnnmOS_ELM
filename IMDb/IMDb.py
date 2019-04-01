import urllib.request
import os
import tarfile
from keras.engine import Model
import tensorflow as tf
from model import OSELM
import numpy as np
#下载数据集
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
# 解压
if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('data/')

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re

re_tag = re.compile(r'<[^>]+>')


def rm_tags(text):
    return re_tag.sub('', text)


import os


def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []

    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts

# 读文件
y_train,train_text=read_files("train")
y_test,test_text=read_files("test")

# 建立单词和数字映射的字典
token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)

#将影评的单词映射到数字
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)

# 让所有影评保持在380个数字
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=380)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model = Sequential()

model.add(Embedding(output_dim=32,
                    input_dim=3800,
                    input_length=380))
model.add(Dropout(0.35))

# 加了一个简单的RNN层
model.add(SimpleRNN(units=16))

model.add(Dense(units=256,activation='relu' ))
model.add(Dropout(0.35))

model.add(Dense(units=1,activation='sigmoid' ))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights("imdb_data.h5")
# train_history =model.fit(x_train, y_train,batch_size=100,
#                          epochs=1,verbose=2,
#                          validation_split=0.2)
# model.save_weights("imdb_data.h5")



def hidden_layer_generate(model):

    """
    CNNの中間層の出力を取得するモデルの構築
    :param cnn_model: CNNモデル
    :return:
    """

    layer_name = 'simple_rnn_1'
    hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    cnn_train_result = hidden_layer_model.predict(x_train)
    cnn_test_result = hidden_layer_model.predict(x_test)
    return hidden_layer_model, cnn_train_result, cnn_test_result



hidden_num = 10
###
batch_size = 100
###


tf.set_random_seed(2016)  #随机序列可重复
sess = tf.Session()

elm = OSELM(sess, batch_size, 16, hidden_num, 1)
#
# data_train_2d, data_test_2d, target_train, target_test = load_mnist_2d()
# # print(target_train.shape)  1203
# cnn_model = cnn_generate(data_train_2d, target_train)

hidden_layer_model, cnn_train_result, cnn_test_result= hidden_layer_generate(model)
# print("cnn_train_result")  #(42500, 10)
# print(cnn_train_result.shape)



epoch = 0
data_container = []
target_container = []
# target_train_convert = y_train
# print("Y_train")   (42500, 10)
# print(Y_train.shape)
# target_test_convert = np_utils.to_categorical(target_test, NUM_CLASS)

#test
# print("X_test")
# print(X_test.shape)
# Y_test = np.array(Y_test)
# print("Y_test")
# print(Y_test.shape)
y_test = np.array(y_test)
print("hidden_num：{},batch_size: {}".format(hidden_num,batch_size))
print("-------------------------------------------------------------------")
while(epoch < 40):
    k = 0
    y_test_use = y_test


    for (index,data) in enumerate(cnn_train_result, start= 0 ):
        if(index >= (epoch) * batch_size and index < (epoch+1) * batch_size):
        # if (index >= (epoch ) * batch_size and index < (epoch +1) * batch_size):
            data_container.append(data)
            k += 1
            if k == batch_size:
                break

    j = 0
    for (index1,target) in enumerate(y_train, start= 0):
        if (index1 >= (epoch) * batch_size and index1 < (epoch + 1) * batch_size):
        # if (index >= (epoch) * batch_size and index < (epoch + 1) * batch_size):
            target_container.append(target)
            j += 1
            if j == batch_size:
                break
    data_container = np.array(data_container)
    target_container = np.array(target_container)
    target_container = target_container[:, np.newaxis]
    elm.train(data_container, target_container)
    # elm.test(data_container, target_container)


    y_test_use = y_test_use[:, np.newaxis]
    elm.test(cnn_test_result, y_test_use)
    # elm.test(X_test, Y_test)

    print(epoch)
    epoch += 1
    data_container = []
    target_container = []