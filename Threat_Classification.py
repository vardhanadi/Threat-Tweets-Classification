import sys

import tensorflow.keras
import sklearn as sk
import tensorflow as tf


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Embedding, MaxPooling1D, Flatten


Train = pd.read_csv("train.csv")
Test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

Train = Train.drop(['keyword', 'location'], axis = 1)
Test = Test.drop(['keyword', 'location'], axis = 1)

print(Train.shape, Test.shape)

# #Visualizing class distribution 
# plt.figure(figsize=(10,5))
# sns.countplot(y='target',data = Train,palette="Paired")
# plt.ylabel("Tweet Fallacy")
# plt.xlabel("Number of tweets")
# plt.show()

# #Visualizing tweet length by words
# plt.figure(figsize=(10,5))
# train_sent = Train['text'].str.split().map(lambda x : len(x))
# sns.boxplot(x="target",y=train_sent,data=Train,palette="Set1")
# plt.xlabel("Tweet Fallacy")
# plt.ylabel("Tweet length by word")
# plt.show()

# fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
# plt.tight_layout()

# Train.groupby('target').count()['id'].plot(kind='pie', ax=axes[0], labels=['Not Disaster (57%)', 'Disaster (43%)'])
# sns.countplot(x=Train['target'], hue=Train['target'], ax=axes[1])

# axes[0].set_ylabel('')
# axes[1].set_ylabel('')
# axes[1].set_xticklabels(['Not Disaster (4342)', 'Disaster (3271)'])
# axes[0].tick_params(axis='x', labelsize=15)
# axes[0].tick_params(axis='y', labelsize=15)
# axes[1].tick_params(axis='x', labelsize=15)
# axes[1].tick_params(axis='y', labelsize=15)

# axes[0].set_title('Target Distribution in Training Set', fontsize=13)
# axes[1].set_title('Target Count in Training Set', fontsize=13)

# plt.show()

#Preparing Data to be pushed in to the model

train_text = Train['text']
y = Train['target']

max_len = 100
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(train_text)
word_index = tokenizer.word_index
len(word_index)

sequences = tokenizer.texts_to_sequences(train_text)
X = pad_sequences(sequences, maxlen=max_len)
X.shape

test_data = tokenizer.texts_to_sequences(Test['text'])
test_data = pad_sequences(test_data, maxlen=max_len)
test_data.shape

y = np.array(y).reshape((-1,1))
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15)

def train_model(model, batch_size=32, epochs=20):
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data = (X_test,y_test) ,validation_split=0.2)
    print('-' * 100)
    print('Test data')
    model.evaluate(X_test, y_test)
    return history


def visual_validation_and_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss'] 
    val_loss = history.history['val_loss']

    epochs_plot = np.arange(1, len(loss) + 1)
    plt.clf()

    plt.plot(epochs_plot, acc, 'r', label='Training acc')
    plt.plot(epochs_plot, val_acc, 'b', label='Validation acc')
    plt.plot(epochs_plot, loss, 'r:', label='Training loss')
    plt.plot(epochs_plot, val_loss, 'b:', label='Validation loss')
    plt.title('Validation and accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

#The Model
model = Sequential([
    Embedding(max_words, 32, input_length=max_len),
    Conv1D(256, 3, activation = 'relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(32, activation = 'sigmoid'),
    Dense(1, activation='sigmoid')])

model.summary()

#Train Model
history = train_model(model)
visual_validation_and_accuracy(history)

#Predict on test_data
predict = model.predict(test_data)
predict.shape

predict = (predict >= 0.5).astype(int)

submission.target = predict
submission.head(10)