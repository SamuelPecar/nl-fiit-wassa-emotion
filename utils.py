import pandas as pd
import numpy as np
import os
import operator
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import contextlib
import sys

tokenizer = Tokenizer(filters='')


def load_dataset(train_file, trial_file, trial_labels, test_file, sep='\t', header=None):
    train = pd.read_csv(train_file, sep=sep, header=header, quoting=3).sample(frac=1).values
    trial = pd.read_csv(trial_file, sep=sep, header=header, quoting=3).values
    trial_label = pd.read_csv(trial_labels, sep=sep, header=header).values
    test = pd.read_csv(test_file, sep=sep, header=header, quoting=3).values

    train_x = np.asarray(train[:, 1])
    train_y = np.asarray(train[:, 0])
    trial_x = np.asarray(trial[:, 1])
    trial_y = np.asarray(trial_label[:, 0])
    test_x = np.asarray(test[:, 1])
    test_y = np.asarray(test[:, 0])

    # split_ratio = int(len(train_x) * 0.9)
    # trial_x = np.concatenate((trial_x, train_x[split_ratio:]))
    # trial_y = np.concatenate((trial_y, train_y[split_ratio:]))
    # train_x = train_x[:split_ratio]
    # train_y = train_y[:split_ratio]

    return train_x, train_y, trial_x, trial_y, test_x, test_y


def create_vocabulary(train_x, trial_x, test_x):
    tokenizer.fit_on_texts(np.concatenate((train_x, trial_x, test_x)))
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
            if j >= max_len:
                break

    return x_indices


def labels_to_indices(y, word_to_index, classes):
    y_indices = []

    for i in range(y.shape[0]):
        y_indices.append(word_to_index[y[i]])

    y_indices = np.asarray(y_indices, dtype='int32')

    return np.eye(classes)[y_indices.reshape(-1)]


def indices_to_labels(y, index_to_word):
    y_classes = []
    for i in range(len(y)):
        y_classes.append(index_to_word[y[i]])
    return y_classes


def load_embeddings(filepath='data/glove.840B.300d.txt'):
    embeddings_index = {}
    with open(os.path.join(filepath), encoding='UTF-8') as f:
        i = 0
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                i += 1
                continue
            embeddings_index[word] = coefs
        if i == 0:
            print("...embeddings loaded")
        else:
            print("...embeddings loaded with {} errors (can be ignored)".format(i))
        return embeddings_index


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


def create_output_csv(y, predictions, probabilities, x, file='data/output.csv'):
    data = pd.DataFrame()

    data.insert(0, 'text', x)
    data.insert(0, 'sad', probabilities[:, 0])
    data.insert(0, 'joy', probabilities[:, 1])
    data.insert(0, 'disgust', probabilities[:, 2])
    data.insert(0, 'surprise', probabilities[:, 3])
    data.insert(0, 'anger', probabilities[:, 4])
    data.insert(0, 'fear', probabilities[:, 5])
    data.insert(0, 'predictions', predictions)
    data.insert(0, 'class', y)
    data.to_csv(file, sep=';')


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


if __name__ == '__main__':
    train_x, _, test_x, _, length = load_dataset('data/train.csv', 'data/trial.csv', 'data/test.labels', partition=None)

    hashtag_dict = dict()

    for x in train_x:
        words = x.split(' ')
        for y in words:
            if y is not '' and not y == '#[#TRIGGERWORD#]' and y[0] == '#':

                if y in hashtag_dict:
                    hashtag_dict[y] += 1
                else:
                    hashtag_dict[y] = 1

    print(hashtag_dict)
    print(len(hashtag_dict.keys()))
    sorted_x = sorted(hashtag_dict.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x)
    zz = {k: v for k, v in hashtag_dict.items() if v > 2}
    print(len(zz.keys()))
    print(zz)
    # plt.figure()
    # plt.bar(range(0,len(hashtag_dict.keys()), 1), hashtag_dict.values(), width=5)
    # plt.show()
