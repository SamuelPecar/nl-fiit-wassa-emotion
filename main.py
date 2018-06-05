import utils
import model_utils

classes = 6
epochs = 20
batch_size = 32
dim = 50
partition = 10000

labels_to_index = {
    "sad": 0,
    "joy": 1,
    "disgust": 2,
    "surprise": 3,
    "anger": 4,
    "fear": 5
}

print('Preparing data')

train_x, train_y, test_x, test_y, max_string_length = utils.load_dataset('data/train.csv', 'data/trial.csv', 'data/test.labels', partition=partition)

vocab_length, words_to_index, index_to_words = utils.create_vocabulary(train_x, test_x)

train_y_oh = utils.labels_to_indices(train_y, labels_to_index, classes)
test_y_oh = utils.labels_to_indices(test_y, labels_to_index, classes)

train_x_indices = utils.sentences_to_indices(train_x, words_to_index, max_len=max_string_length)
test_x_indices = utils.sentences_to_indices(test_x, words_to_index, max_len=max_string_length)

print('Creating embedding layer')

# word_embeddings = utils.load_embeddings()
# embeddings_layer = model_utils.create_embedding_layer(word_embeddings, words_to_index, len(words_to_index), output_dim=dim)
embeddings_layer = {}

print('Creating model')

model = model_utils.get_model((max_string_length,), embeddings_layer, vocab_length, classes)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = model_utils.get_callbacks()
model_info = model.fit(train_x_indices, train_y_oh, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks, shuffle=True)
utils.plot_model_history(model_info)

print('predict values model')

loss, acc = model.evaluate(test_x_indices, test_y_oh)

print('evaluate model')

print("Loss = ", loss)
print("Test accuracy = ", acc)
