import csv
import numpy as np
import keras as kr
import tensorflow as tf
from model import OSELM
from keras.engine import Model

import datetime


# Load the Iris dataset.
# Data from: https://github.com/mwaskom/seaborn-data/blob/master/iris.csv
iris = list(csv.reader(open('iris.csv')))[1:]

# The inputs are four floats: sepal length, sepal width, petal length, petal width.
inputs  = np.array(iris)[:,:4].astype(np.float)

# Outputs are initially individual strings: setosa, versicolor or virginica.
outputs = np.array(iris)[:,4]
# Convert the output strings to ints.
outputs_vals, outputs_ints = np.unique(outputs, return_inverse=True)
# Encode the category integers as binary categorical vairables.
outputs_cats = kr.utils.to_categorical(outputs_ints)

# Split the input and output data sets into training and test subsets.
inds = np.random.permutation(len(inputs))
train_inds, test_inds = np.array_split(inds, 2)
inputs_train, outputs_train = inputs[train_inds], outputs_cats[train_inds]
print(inputs_train.shape)
inputs_test,  outputs_test  = inputs[test_inds],  outputs_cats[test_inds]

# Create a neural network.
model = kr.models.Sequential()

# Add an initial layer with 4 input nodes, and a hidden layer with 16 nodes.
model.add(kr.layers.Dense(16, input_shape=(4,)))
# Apply the sigmoid activation function to that layer.
model.add(kr.layers.Activation("sigmoid"))
# Add another layer, connected to the layer with 16 nodes, containing three output nodes.
model.add(kr.layers.Dense(3))
# Use the softmax activation function there.
model.add(kr.layers.Activation("softmax"))

# Configure the model for training.
# Uses the adam optimizer and categorical cross entropy as the loss function.
# Add in some extra metrics - accuracy being the only one.
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the model using our training data.
# starttime = datetime.datetime.now()

model.load_weights("iris.h5")

# model.fit(inputs_train, outputs_train, epochs=50, batch_size=1, verbose=1)



#
# endtime = datetime.datetime.now()
#
# print (endtime - starttime)
model.summary()
# Evaluate the model using the test data set.
loss, accuracy = model.evaluate(inputs_train, outputs_train, verbose=1)
#
# # Output the accuracy of the model.
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))
# #
# # Predict the class of a single flower.
# prediction = np.around(model.predict(np.expand_dims(inputs_test[0], axis=0))).astype(np.int)[0]
# print("Actual: %s\tEstimated: %s" % (outputs_test[0].astype(np.int), prediction))
# print("That means it's a %s" % outputs_vals[prediction.astype(np.bool)][0])




#
# # Save the model to a file for later use.
# model.save("iris_nn.h5")
# # Load the model again with: model = load_model("iris_nn.h5")


def hidden_layer_generate(model):

    """
    CNN¤ÎÖÐégŒÓ¤Î³öÁ¦¤òÈ¡µÃ¤¹¤ë¥â¥Ç¥ë¤Î˜‹ºB
    :param cnn_model: CNN¥â¥Ç¥ë
    :return:
    """

    layer_name = 'activation_1'
    hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    cnn_train_result = hidden_layer_model.predict(inputs_train)
    cnn_test_result = hidden_layer_model.predict(inputs_test)
    return hidden_layer_model, cnn_train_result, cnn_test_result



hidden_num = 5
###
batch_size = 10
###


tf.set_random_seed(2016)
sess = tf.Session()

elm = OSELM(sess, batch_size, 16, hidden_num, 3)
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

# print("Y_train")   (42500, 10)
# print(Y_train.shape)
# target_test_convert = np_utils.to_categorical(target_test, NUM_CLASS)

#test
# print("X_test")
# print(X_test.shape)
# Y_test = np.array(outputs_train)
# print("Y_test")
# print(Y_test.shape)

print("hidden_num£º{},batch_size: {}".format(hidden_num,batch_size))
print("-------------------------------------------------------------------")
starttime = datetime.datetime.now()
while(epoch < 7):
    k = 0

    for (index,data) in enumerate(cnn_train_result, start= 0 ):
        if(index >= (epoch) * batch_size and index < (epoch+1) * batch_size):
        # if (index >= (epoch ) * batch_size and index < (epoch +1) * batch_size):
            data_container.append(data)
            k += 1
            if k == batch_size:
                break

    j = 0
    for (index1,target) in enumerate(outputs_train, start= 0):
        if (index1 >= (epoch) * batch_size and index1 < (epoch + 1) * batch_size):
        # if (index >= (epoch) * batch_size and index < (epoch + 1) * batch_size):
            target_container.append(target)
            j += 1
            if j == batch_size:
                break
    data_container = np.array(data_container)
    target_container = np.array(target_container)
    elm.train(data_container, target_container)
    # elm.test(X_test, Y_test)
    # elm.test(cnn_test_result, outputs_test)
    elm.test(data_container, target_container)

    print(epoch)
    epoch += 1
    data_container = []
    target_container = []
endtime = datetime.datetime.now()
print (endtime - starttime)
# elm.test(cnn_train_result, Y_test)