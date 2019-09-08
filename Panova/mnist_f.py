from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
import matplotlib.pyplot as plt

#load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#normalize data
train_images = train_images/255.0
test_images = test_images/255.0

'''
#show numbers from dataset
plt.figure(figsize=(7,9))
for i in range(15):
    plt.subplot(5,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.savefig('mnist.pdf')
plt.show()
'''

def model():
    #create the model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#build the model
model = model()

#fit the model
model.fit(train_images, train_labels, epochs=10)

#accuracy estimation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Accuracy:', test_acc)


"""Predictions visualization"""

predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, image):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image[i], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array[i])

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100*np.max(predictions_array[i]), true_label[i]))

def plot_prediction(i, predictions_array, true_label):
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array[i], color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array[i])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label[i]].set_color('blue')

#plot recognised numbers
plt.figure(figsize=(9, 7.5))
for i in range(15):
    plt.subplot(5, 2*3, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(5, 2*3, 2*i+2)
    plot_prediction(i, predictions, test_labels)
plt.savefig('mnist_res.pdf')
plt.show()

#plot numbers that were recognised incorrectly
plt.figure(figsize=(9, 7.5))
start = 0
for i in range(15):
    for j in range(start, 10000):
        if np.argmax(predictions[j]) != test_labels[j]:
            plt.subplot(5, 2*3, 2*i+1)
            plot_image(j, predictions, test_labels, test_images)
            plt.subplot(5, 2*3, 2*i+2)
            plot_prediction(j, predictions, test_labels)
            start = j+1
            break
plt.savefig('mnist_false.pdf')
plt.show()