import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Data Load
data=keras.datasets.fashion_mnist

# Split data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# View
#plt.imshow(train_images[7])
#plt.show()

# Normalize
train_images = train_images/255.0
test_images = test_images/255.0
#input,hidden, output
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])
# Load parameters in model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#Training(epochs=times that use one same image more epochs = menos reliable)
model.fit(train_images, train_labels, epochs=5)
# Testing model
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print('\nTest accuracy:', test_acc)
# Multiple
prediction = model.predict(test_images)
# One
#prediction = model.predict([test_images[7],])

#print(class_names(np.argmax(prediction[0])))

plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()