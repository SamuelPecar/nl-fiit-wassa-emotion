import sys
import utils
import model_utils
import preprocessing
import evaluation
import config
import slack

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

print('Preparing data')

train_x, train_y, trial_x, trial_y, test_x, test_y = utils.load_dataset('data/train.csv', 'data/trial.csv', 'data/trial.labels', 'data/test.csv', partition=config.partition)


print('Preprocessing data')
train_x, trial_x, test_x, max_string_length = preprocessing.preprocessing_pipeline(train_x, trial_x, test_x, emoji2word=config.emoji2word)

vocab_length, words_to_index, index_to_words = utils.create_vocabulary(train_x, trial_x, test_x)

train_y_oh = utils.labels_to_indices(train_y, config.labels_to_index, config.classes)
trial_y_oh = utils.labels_to_indices(trial_y, config.labels_to_index, config.classes)

train_x_indices = utils.sentences_to_indices(train_x, words_to_index, max_len=max_string_length)
trial_x_indices = utils.sentences_to_indices(trial_x, words_to_index, max_len=max_string_length)
test_x_indices = utils.sentences_to_indices(test_x, words_to_index, max_len=max_string_length)

print('Creating embedding layer')

word_embeddings = utils.load_embeddings(filepath=config.embeddings_path)
embeddings_layer = model_utils.create_embedding_layer(word_embeddings, words_to_index, len(words_to_index), output_dim=config.dim)

print('Creating model')
model = model_utils.get_model((max_string_length,), embeddings_layer, config.classes, units=config.units)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = model_utils.get_callbacks(early_stop_monitor=config.early_stop_monitor, early_stop_patience=config.early_stop_patience, early_stop_mode=config.early_stop_mode)

weights = model_utils.get_sample_weights_prim(train_y, config.class_weight)

# model_info = model.fit(train_x_indices, train_y_oh, epochs=config.epochs, batch_size=config.batch_size, validation_split=0.05, callbacks=callbacks, shuffle=True, verbose=config.verbose)
model_info = model.fit(train_x_indices, train_y_oh, epochs=config.epochs, batch_size=config.batch_size, validation_data=(trial_x_indices, trial_y_oh), callbacks=callbacks, shuffle=True, verbose=config.verbose, sample_weight=weights)

print('predict values model')

probabilities_trial = model.predict(trial_x_indices)
predictions_trial = utils.indices_to_labels(probabilities_trial.argmax(axis=-1), config.index_to_label)
microaverage, macroaverage = evaluation.calculate_prf(trial_y.tolist(), predictions_trial)
utils.create_output_csv(trial_y, predictions_trial, probabilities_trial, trial_x, file='data/trial_results.csv')

probabilities = model.predict(test_x_indices)
predictions = utils.indices_to_labels(probabilities.argmax(axis=-1), config.index_to_label)

# microaverage, macroaverage = evaluation.calculate_prf(trial_y_oh.tolist(), predictions)

utils.create_output_csv(None, predictions, probabilities, test_x, file='data/test_results.csv')

token = "xoxp-18602746578-256894923987-385376204052-7892a6d3375c5c19af86d57f6c75b92e"
slack.slack_message(microaverage, 'iest', token)
slack.slack_message(macroaverage, 'iest', token)
