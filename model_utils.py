from keras.models import Model
from keras.layers import LSTM, Dense, Input, Dropout, Activation, Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping
import numpy as np

# val_f1_c
def get_callbacks(early_stop_monitor='val_acc'):
    earlystop = EarlyStopping(monitor=early_stop_monitor, min_delta=0.001, patience=4, verbose=1, mode='auto')

    return [earlystop]


def create_embedding_layer(embeddings_index, words_to_index, vocab_length=100, output_dim=300):
    emb_matrix = np.zeros((vocab_length + 1, output_dim))

    for word, index in words_to_index.items():
        if word in embeddings_index:
            emb_matrix[index, :] = embeddings_index[word]

    embedding_layer = Embedding(vocab_length + 1, output_dim, trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def get_model(input_shape, embedding_layer, classes=6):
    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = embedding_layer(sentence_indices)

    x = Bidirectional(LSTM(units=128, return_sequences=False))(embeddings)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=classes, activation="softmax")(x)

    return Model(inputs=sentence_indices, outputs=x)
