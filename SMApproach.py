# -*- coding: utf-8 -*-
import keras
import tensorflow
import numpy
import pandas as pd
import torch
import utils
import preprocessing
import csv
import tensorflow_hub as hub
import model_utils
import evaluation
import config


class AbstractEncoder():
    def __init__(self, args):
        print("...Initializing encoder of type {}".format(type(self)))
    def use(self, sentences_to_embed):
        print("Not implemented!")
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
    def __init__(self, all_sentences=None):
        self.enc = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.session = tensorflow.Session()
        self.session.run([tensorflow.global_variables_initializer(), tensorflow.tables_initializer()])
        self.sentence_ph = tensorflow.placeholder(tensorflow.string, shape=(None))
        self.encodings = self.enc(self.sentence_ph)
    def use(self, sentences_to_embed):
        return self.session.run(self.encodings, feed_dict={self.sentence_ph: sentences_to_embed})
    @staticmethod
    def getName():
        return "USE"

class Infersent(AbstractEncoder):
    def __init__(self, all_sentences):
        self.infersent = torch.load('SentenceModels/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
        self.infersent.set_glove_path('data/glove.840B.300d.txt')

        with utils.nostdout():
            self.infersent.build_vocab(all_sentences)
    def use(self, sentences_to_embed):
        definitions_emb = list()
        for definition in sentences_to_embed:
            definitions_emb.append(self.infersent.encode([definition])[0])

        return definitions_emb
    @staticmethod
    def getName():
        return "InferSent"

def prepare_data_file(encoder, dataset_size=10000, batch_size=120):
    train_x, train_y, test_x, test_y = utils.load_dataset('data/train.csv', 'data/trial.csv', 'data/test.labels', partition=dataset_size)
    train_x, test_x, max_string_length = preprocessing.preprocessing_pipeline(train_x, test_x, emoji2word=True)

    train_x_list = list()
    train_y_list = list()

    for index, sentence in enumerate(train_x):
        train_y_list.append(train_y[index])
        train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'angry'))
        train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'sad'))
        train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'joy'))
        train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'disgust'))
        train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'fear'))
        train_x_list.append(sentence.replace("[#TRIGGERWORD#]", 'surprise'))
    encoder_instance = encoder(train_x_list)


    with open("data/enRep_{}_{}.csv".format(encoder.getName(), dataset_size), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        i=0
        while(True):
            print('\r.........{0}/{1}'.format(i*batch_size, len(train_x_list)), end="")
            embeddings = encoder_instance.use(train_x_list[i*batch_size:(i+1)*batch_size])
            for x in range(0, batch_size, 6):
                new_row = train_y_list[i]
                for j in range(0, 6):
                    new_row += "," + numpy.array_str(embeddings[x+j]).replace('\n', '')
                csv_writer.writerow([new_row])
            if (i+1)*batch_size >= len(train_x_list):
                i += 1
                print('\r.........{0}/{1}'.format(i * batch_size, len(train_x_list)), end="")
                break
            else:
                i += 1

def load_data(filename, partition=None):
    train = pd.read_csv(filename, sep=',').sample(frac=1).values
    if not partition:
        partition = train.shape[0]
    train_x = numpy.zeros((partition, train.shape[1]-1, numpy.fromstring(train[0][1][1:-1], dtype=numpy.float32, sep=' ').shape[0]))
    train_y = numpy.zeros((partition, 6))
    print(train_x.shape)
    for i in range(0, partition):
        train_x[i, 0] = numpy.fromstring(train[i, 1][1:-1], dtype=numpy.float32, sep=' ')
        train_x[i, 1] = numpy.fromstring(train[i, 2][1:-1], dtype=numpy.float32, sep=' ')
        train_x[i, 2] = numpy.fromstring(train[i, 3][1:-1], dtype=numpy.float32, sep=' ')

        train_x[i, 3] = numpy.fromstring(train[i, 4][1:-1], dtype=numpy.float32, sep=' ')
        train_x[i, 4] = numpy.fromstring(train[i, 5][1:-1], dtype=numpy.float32, sep=' ')
        train_x[i, 5] = numpy.fromstring(train[i, 6][1:-1], dtype=numpy.float32, sep=' ')

        if train[i,0] == 'angry':
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


    return train_x, train_y
def experiment1(encoder):
    train_x, train_y = load_data("data/enRep_USE_None.csv", 40000)

    model = model_utils.get_SM_model(train_x[0].shape)

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', evaluation.f1, evaluation.precision, evaluation.recall])
    callbacks = model_utils.get_callbacks(early_stop_monitor=config.early_stop_monitor,
                                          early_stop_patience=config.early_stop_patience,
                                          early_stop_mode=config.early_stop_mode)
    model_info = model.fit(train_x, train_y, epochs=config.epochs, batch_size=config.batch_size,
                           validation_split=0.1, shuffle=True, verbose=2)

if __name__ == '__main__':
    #experiment1(Infersent)
    #prepare_data_file(Infersent)
    prepare_data_file(UniversalSentenceEncoder, 200)






