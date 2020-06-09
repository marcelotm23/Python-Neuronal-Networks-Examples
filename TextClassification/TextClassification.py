import tensorflow as tf
from tensorflow import keras
import numpy

# Data load about movies
data = keras.datasets.imdb

# Split data, num_words specify the number of words more frecuent
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

# A dictionary mapping words to an integer index
_word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Preprocessing Data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# this function will return the decoded (human readable) reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


"""
# Defining model
model = keras.Sequential()
# Embedded layer
model.add(keras.layers.Embedding(88000, 16))#88000 word vectors, 16 dimensions
# Average layer
model.add(keras.layers.GlobalAveragePooling1D())
# Dense layer
model.add(keras.layers.Dense(16, activation="relu"))#rectifier linear unit
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model
#print(decode_review(train_data[0]))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Validation data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Training the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# Testing the model
results = model.evaluate(test_data, test_labels)

# Saving the model
model.save("model.h5")
"""
# Loading the model
model = keras.models.load_model("model.h5")


# Transforming our data
def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # Remove symbols to split by space and encode
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"",
                                                                                                                  "").strip().split(
            " ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)  # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

# Analysis

test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
#print(results)
