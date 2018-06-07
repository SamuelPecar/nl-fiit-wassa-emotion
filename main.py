import utils
import model_utils
import preprocessing
import evaluation

classes = 6
epochs = 20
batch_size = 64
dim = 25
# partition = 1000
partition = None
embeddings_path = 'data/glove.twitter.27B.25d.txt'
labels_to_index = {
    "sad": 0,
    "joy": 1,
    "disgust": 2,
    "surprise": 3,
    "anger": 4,
    "fear": 5
}

print('Preparing data')

train_x, train_y, test_x, test_y = utils.load_dataset('data/train.csv', 'data/trial.csv', 'data/test.labels', partition=partition)

print('Preprocessing data')
train_x, test_x, max_string_length = preprocessing.preprocessing_pipeline(train_x, test_x, emoji2word=False)

vocab_length, words_to_index, index_to_words = utils.create_vocabulary(train_x, test_x)

train_y_oh = utils.labels_to_indices(train_y, labels_to_index, classes)
test_y_oh = utils.labels_to_indices(test_y, labels_to_index, classes)

train_x_indices = utils.sentences_to_indices(train_x, words_to_index, max_len=max_string_length)
test_x_indices = utils.sentences_to_indices(test_x, words_to_index, max_len=max_string_length)

print('Creating embedding layer')

word_embeddings = utils.load_embeddings(filepath=embeddings_path)
embeddings_layer = model_utils.create_embedding_layer(word_embeddings, words_to_index, len(words_to_index), output_dim=dim)

print('Creating model')

model = model_utils.get_model((max_string_length,), embeddings_layer, classes)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[['accuracy', evaluation.f1_c, evaluation.precision, evaluation.recall]])
callbacks = model_utils.get_callbacks()
model_info = model.fit(train_x_indices, train_y_oh, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks, shuffle=True)
utils.plot_model_history(model_info)

print('predict values model')

loss, acc = model.evaluate(test_x_indices, test_y_oh)

print('evaluate model')

print("Loss = ", loss)
print("Test accuracy = ", acc)
