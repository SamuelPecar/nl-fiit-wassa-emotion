# -*- coding: utf-8 -*-
import keras
import tensorflow
import numpy
import pandas as pd
import torch
import utils
import modules.preprocessing as preprocessing
import csv
import tensorflow_hub as hub
import model_utils
import modules.evaluation as evaluation
import config
import torch
import generators

class AbstractEncoder():
    def __init__(self, args):
        print("...Initializing encoder of type {}".format(type(self)))
    def use(self, sentences_to_embed):
        print("Not implemented!")
    def getLambda(self):
        raise NotImplementedError
# class Sent2Vec(AbstractEncoder):
#     def __init__(self):
#         self.model = sent2vec.Sent2vecModel()
#         s2v_path = '/home/mehre/[DRIVE]/[Dev]/[AI]/sent2vec-master'
#         self.model.load_model(s2v_path+'/torontobooks_unigrams.bin')
#
#     def use(self, sentences_to_embed):
#         definitions_emb = list()
#         for definition in sentences_to_embed:
#             definitions_emb.append(self.model.embed_sentence(definition))
#
#         return definitions_emb
# class Skip_Thoughts(AbstractEncoder):
#     def __init__(self):
#         st_path = '/home/mehre/[DRIVE]/[Dev]/[AI]/skip-thoughts-master/'
#         skipthoughts.path_to_models = st_path
#         skipthoughts.path_to_tables = st_path
#         skipthoughts.path_to_umodel = st_path + 'uni_skip.npz'
#         skipthoughts.path_to_bmodel = st_path + 'bi_skip.npz'
#         model = skipthoughts.load_model()
#         self.encoder = skipthoughts.Encoder(model)
#
#     def use(self, sentences_to_embed):
#         definitions_emb = list()
#         for definition in sentences_to_embed:
#             definitions_emb.append(self.encoder.encode([definition])[0])
#
#         return definitions_emb
class UniversalSentenceEncoder(AbstractEncoder):
    def __init__(self, all_sentences=None, type="large"):
        if type == "large":
            self.enc = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        elif type == "small":
            self.enc = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        #self.session = tensorflow.Session()
        #self.session.run([tensorflow.global_variables_initializer(), tensorflow.tables_initializer()])
        #self.sentence_ph = tensorflow.placeholder(tensorflow.string, shape=(None))
        #self.encodings = self.enc(self.sentence_ph)
    def use(self, sentences_to_embed):
        return self.session.run(self.encodings, feed_dict={self.sentence_ph: sentences_to_embed})
    def use_lambda(self, x):
        return self.enc(inputs=tensorflow.squeeze(tensorflow.cast(x, tensorflow.string)))
    def getLambda(self, shape):
        sess = tensorflow.Session()
        keras.backend.set_session(sess)
        sess.run(tensorflow.global_variables_initializer())
        sess.run(tensorflow.tables_initializer())
        return keras.layers.Lambda(self.use_lambda, output_shape=shape)
    @staticmethod
    def getName():
        return "USE"
    @staticmethod
    def getSize():
        return 512

class Infersent(AbstractEncoder):
    def __init__(self, all_sentences):
        #self.infersent = torch.load('SentenceModels/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
        self.infersent = torch.load('SentenceModels/infersent.allnli.pickle')
        self.infersent.set_glove_path('data/glove.840B.300d.txt')

        with utils.nostdout():
            self.infersent.build_vocab(all_sentences)
    def use(self, sentences_to_embed):
        definitions_emb = list()
        for definition in sentences_to_embed:
            definitions_emb.append(self.infersent.encode([definition], tokenize=True)[0])

        return definitions_emb

    def use_lambda(self, sentences_to_embed):
        # definitions_emb = numpy.zeros((config.batch_size, self.getSize()))
        # for index in range(0, config.batch_size):
        #     definitions_emb[index] = self.infersent.encode([sentences_to_embed[index]], tokenize=True)

        return self.infersent.encode([sentences_to_embed],tokenize=True)

    def getLambda(self):
        return keras.layers.Lambda(self.use_lambda)
    @staticmethod
    def getName():
        return "InferSent"
    @staticmethod
    def getSize():
        return 4096



def prepare_data_file(encoder, dataset_size=10000, batch_size=128, mode='original'):  #original, reconstructed, both
    print("...loading&preprocessing dataset")
    train_x, train_y, test_x, test_y, _, _ = utils.load_dataset('data/train_ekphrasis.csv', 'data/trial_ekphrasis.csv', 'data/trial.labels', 'data/test_ekphrasis.csv')
    train_x = train_x[:dataset_size]
    train_y = train_y[:dataset_size]
    train_x, test_x, _,max_string_length = preprocessing.preprocessing_pipeline(train_x, test_x, _)
    print("...preparing candidate sentences:")
    train_x_list = list()
    train_y_list = list()

    for index, sentence in enumerate(train_x):
        print('\r......{0}/{1}'.format(index, train_x.shape[0]), end="")
        train_y_list.append(train_y[index])
        if mode == 'original' or mode == 'both':
            train_x_list.append(sentence)
        if mode == 'reconstructed' or mode == 'both':
            train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'angry'))
            train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'sad'))
            train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'joy'))
            train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'disgust'))
            train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'fear'))
            train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'surprise'))
    print("\n...initializing encoder instance")
    encoder_instance = encoder(train_x_list)
    numpy.set_printoptions(threshold=5000)
    print("...writing csv:")
    if mode == 'both':
        n_of_elements_in_row = 7
    elif mode == 'original':
        n_of_elements_in_row = 1
    elif mode == 'reconstructed':
        n_of_elements_in_row = 6
    else:
        raise ValueError
    with open("data/enRep_{}_{}_{}.csv".format(encoder.getName(), dataset_size, mode), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        i = 0
        j = 0
        while(True):
            print('\r.........{0}/{1}'.format(i*batch_size, len(train_x_list)), end="")
            embeddings = encoder_instance.use(train_x_list[i*batch_size:(i+1)*batch_size])
            for x in range(0, len(embeddings)):
                if x % n_of_elements_in_row == 0:
                    new_row = train_y_list[j]
                    j += 1
                new_row += "," + numpy.array_str(embeddings[x]).replace('\n', '') + "," + train_x_list[i*batch_size+x]
                if x % n_of_elements_in_row == n_of_elements_in_row-1:
                    csv_writer.writerow([new_row])
            if (i+1)*batch_size >= len(train_x_list):
                i += 1
                print('\r.........{0}/{1}'.format(len(train_x_list), len(train_x_list)), end="")
                break
            else:
                i += 1

def load_data(filename, partition=None, candidates=True):
    #train = pd.read_csv(filename, sep=',', header=None).values
    train = pd.read_csv(filename, sep=',', header=None, nrows=partition).sample(frac=1).values
    if not partition:
        partition = train.shape[0]
    if candidates:
        train_x = numpy.zeros((partition, train.shape[1] - 2, numpy.fromstring(train[0][1][1:-1], dtype=numpy.float32, sep=' ').shape[0]), dtype=numpy.float32)
    else:
        train_x = numpy.zeros((partition, numpy.fromstring(train[0][1][1:-1].replace('\n', ''), dtype=numpy.float32, sep=' ').shape[0]), dtype=numpy.float32)
    train_y = numpy.zeros((partition, 6))
    print(train_x.shape)
    for i in range(0, partition):
        if candidates:
            train_x[i, 0] = numpy.fromstring(train[i, 2][1:-1], dtype=numpy.float32, sep=' ')
            train_x[i, 1] = numpy.fromstring(train[i, 3][1:-1], dtype=numpy.float32, sep=' ')
            train_x[i, 2] = numpy.fromstring(train[i, 4][1:-1], dtype=numpy.float32, sep=' ')

            train_x[i, 3] = numpy.fromstring(train[i, 5][1:-1], dtype=numpy.float32, sep=' ')
            train_x[i, 4] = numpy.fromstring(train[i, 6][1:-1], dtype=numpy.float32, sep=' ')
            train_x[i, 5] = numpy.fromstring(train[i, 7][1:-1], dtype=numpy.float32, sep=' ')
        else:
            train_x[i] = numpy.fromstring(train[i, 1][1:-1], dtype=numpy.float32, sep=' ')

        if train[i,0] == 'anger':
            train_y[i] = numpy.asarray([1, 0, 0, 0, 0, 0])
        elif train[i,0] == 'sad':
            train_y[i] = numpy.asarray([0, 1, 0, 0, 0, 0])
        elif train[i, 0] == 'joy':
            train_y[i] = numpy.asarray([0, 0, 1, 0, 0, 0])
        elif train[i, 0] == 'disgust':
            train_y[i] = numpy.asarray([0, 0, 0, 1, 0, 0])
        elif train[i, 0] == 'fear':
            train_y[i] = numpy.asarray([0, 0, 0, 0, 1, 0])
        elif train[i, 0] == 'surprise':
            train_y[i] = numpy.asarray([0, 0, 0, 0, 0, 1])

    print(train_x[0])
    return train_x, train_y
def experiment_1(encoder):
    print("...loading&preprocessing dataset")
    #train_x, train_y = load_data("data/enRep_USE_None.csv", 40000)
    train_x, train_y, test_x, test_y, _, _ = utils.load_dataset('data/train.csv', 'data/trial.csv',
                                                                'data/trial.labels', 'data/test.csv')
    train_x, test_x, _, max_string_length = preprocessing.preprocessing_pipeline(train_x, test_x, _)
    print("...creating encoder instance")
    encoder_instance = encoder(train_x)
    print("...encoding sentences")
    embeddings = numpy.zeros((train_x.shape[0], encoder_instance.getSize()))
    for i in range(0, train_x.shape[0], 1000):
        print('\r......{0}/{1}'.format(i, train_x.shape[0]), end="")
        embeddings[i:i+1000] = encoder_instance.use(train_x[i:i+1000])
    for i in embeddings[0:10]:
        print(i)
    print("...preparing labels")
    train_y_l = numpy.zeros((train_y.shape[0], 6))
    for i in range(train_y.shape[0]):
        if train_y[i] == 'anger':
            train_y_l[i] = numpy.asarray([1, 0, 0, 0, 0, 0])
        elif train_y[i] == 'sad':
            train_y_l[i] = numpy.asarray([0, 1, 0, 0, 0, 0])
        elif train_y[i] == 'joy':
            train_y_l[i] = numpy.asarray([0, 0, 1, 0, 0, 0])
        elif train_y[i] == 'disgust':
            train_y_l[i] = numpy.asarray([0, 0, 0, 1, 0, 0])
        elif train_y[i] == 'fear':
            train_y_l[i] = numpy.asarray([0, 0, 0, 0, 1, 0])
        elif train_y[i] == 'surprise':
            train_y_l[i] = numpy.asarray([0, 0, 0, 0, 0, 1])
    model = model_utils.get_SM_model_2(embeddings[0].shape, 6)
    print("...Preparations complete:")
    print("......Embeddings shape (data size, embedding size): {}".format(embeddings.shape))
    print("......Labels shape (data size, classes): {}".format(train_y.shape))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', evaluation.f1, evaluation.precision, evaluation.recall])
    callbacks = model_utils.get_callbacks(early_stop_monitor=config.early_stop_monitor,
                                          early_stop_patience=config.early_stop_patience,
                                          early_stop_mode=config.early_stop_mode)
    model_info = model.fit(embeddings, train_y_l, epochs=200, batch_size=config.batch_size,
                           validation_split=0.1, shuffle=True, verbose=2)

def prepare_testdata():
    print("...Preparing test set")

    test = pd.read_csv('data/trial.csv', sep='\t', header=None, quoting=3).values
    test_label = pd.read_csv('data/trial.labels', sep="\t", header=None, quoting=3).values

    test_x = numpy.asarray(test[:, 1])
    test_label = numpy.asarray(test_label[:, 0])

    test_x = preprocessing.preprocessing_pipeline([], [], test_x)[2]
    test_y = utils.labels_to_indices(test_label, config.labels_to_index, 6)

    encoder = Infersent(test_x)
    test_x_s = numpy.zeros((test_x.shape[0], encoder.getSize()))

    i=0
    while(True):
        print('\r......{0}/{1}'.format(i, test_x.shape[0]), end="")
        test_x_s[i:i+64] = encoder.use(test_x[i:i+64])
        i += 64
        if i >= test_x.shape[0]:
            break

    # for index, sentence in enumerate(test_x):
    #     print('\r......{0}/{1}'.format(index, test_x.shape[0]), end="")
    #
    #     test_x_s[index] = encoder.use([sentence])[0]

    return test_x_s, test_y, test_label
def experiment_2(): # w/o candidates
    test_x_s, test_y, test_label = prepare_testdata()
    print("...initializing model")
    model = model_utils.get_SM_model_2((4096,), embedding_layer=None, units=config.units)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    callbacks = model_utils.get_callbacks(early_stop_monitor=config.early_stop_monitor,
                                          early_stop_patience=config.early_stop_patience,
                                          early_stop_mode=config.early_stop_mode)

    model_info = model.fit_generator(generators.MySMGenerator("data/enRep_InferSent_None_original.csv", batch=32, embedding_size=4096), validation_data= generators.MySMValidationGenerator("data/enRep_InferSent_None_original.csv", batch=32, embedding_size=4096), epochs=40,callbacks=callbacks, verbose=2)

    print('...Evaluation')
    loss, acc = model.evaluate(test_x_s, test_y, verbose=2)
    print("...Loss = ", loss)
    print("...Test accuracy = ", acc)

    probabilities = model.predict(test_x_s)
    predictions = utils.indices_to_labels(probabilities.argmax(axis=-1), config.index_to_label)

    evaluation.calculate_prf(test_label.tolist(), predictions)


if __name__ == '__main__':
    # if you want to run experiments with InferSent first prepare data_file and then run experiment_2
    experiment_2()
    #prepare_data_file(Infersent, None, mode='original')







