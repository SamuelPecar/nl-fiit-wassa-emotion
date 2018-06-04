from keras.models import Model
from keras.layers import LSTM, Dense, Input, Dropout, Activation, Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping


def get_callbacks():
    earlystop = EarlyStopping(monitor='acc', min_delta=0.001, patience=5, verbose=1, mode='auto')

    return [earlystop]


def get_model(input_shape, embedding_layer, vocab_length, classes=6):
    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = Embedding(vocab_length + 1, 128, input_length=input_shape[0])(sentence_indices)

    x = Bidirectional(LSTM(units=128, return_sequences=False))(embeddings)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=classes, activation="softmax")(x)
    x = Activation('softmax')(x)

    return Model(inputs=sentence_indices, outputs=x)

# def get_model(input_shape, embedding_layer, vocab_length, classes=3):
#     sentence_indices = Input(shape=input_shape, dtype='int32')
#
#     embeddings = embedding_layer(sentence_indices)
#
#     x = Bidirectional(LSTM(units=128, return_sequences=False))(embeddings)
#     x = Dropout(rate=0.5)(x)
#
#     x = Bidirectional(LSTM(units=128))(x)
#     x = Dropout(rate=0.5)(x)
#
#     x = Dense(units=classes, activation="softmax")(x)
#     x = Activation('softmax')(x)
#
#     return Model(inputs=sentence_indices, outputs=x)
