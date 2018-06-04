import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot

tokenizer = Tokenizer()


def load_dataset(train_file, test_file, test_labels, sep='\t', header=None):
    train = pd.read_csv(train_file, sep=sep, header=header).values
    test = pd.read_csv(test_file, sep=sep, header=header).values
    test_label = pd.read_csv(test_labels, sep=sep, header=header).values

    train_x = np.asarray(train[:10000, 1])
    train_y = np.asarray(train[:10000, 0])

    test_x = np.asarray(test[:1000, 1])
    test_y = np.asarray(test_label[:1000, 0])

    return train_x, train_y, test_x, test_y, len(max(train_x, key=len).split())


def create_vocabulary(train_x, test_x):
    tokenizer.fit_on_texts(np.concatenate((train_x, test_x)))
    words_to_index = tokenizer.word_index
    index_to_words = {v: k for k, v in words_to_index.items()}

    return len(words_to_index), words_to_index, index_to_words


def sentences_to_indices(x, word_to_index, max_len):
    m = x.shape[0]

    x_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = x[i].lower().split()
        j = 0

        for w in sentence_words:
            if w in word_to_index:
                x_indices[i, j] = word_to_index[w]
            else:
                x_indices[i, j] = 0
            j = j + 1

    return x_indices


def labels_to_indices(y, word_to_index, classes):
    y_indices = []

    for i in range(y.shape[0]):
        y_indices.append(word_to_index[y[i]])

    y_indices = np.asarray(y_indices, dtype='int32')

    return np.eye(classes)[y_indices.reshape(-1)]


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
