import tensorflow as tf
from tensorflow import keras
import numpy

# Data load about movies
data=keras.datasets.imdb

#Split data, num_words specify the number of words more frecuent
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

# A dictionary mapping words to an integer index
_word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Preprocessing Data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Defining model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))#rectifier linear unit
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

# this function will return the decoded (human readable) reviews
def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(train_data[0]))
