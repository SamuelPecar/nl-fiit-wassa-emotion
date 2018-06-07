# -*- coding: utf-8 -*-
import sent2vec
import skipthoughts
import keras
import tensorflow
import numpy
import pandas
import torch
import utils
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class AbstractEncoder():
    def __init__(self, args):
        print("...Initializing encoder of type {}".format(type(self)))
    def use(self, word_def, sim_word_def, ant_word_def):
        print("Not implemented!")
class Sent2Vec(AbstractEncoder):
    def __init__(self):
        self.model = sent2vec.Sent2vecModel()
        s2v_path = '/home/mehre/[DRIVE]/[Dev]/[AI]/sent2vec-master'
        self.model.load_model(s2v_path+'/torontobooks_unigrams.bin')

    def use(self, sentences_to_embed):
        definitions_emb = list()
        for definition in sentences_to_embed:
            definitions_emb.append(self.model.embed_sentence(definition))

        return definitions_emb
class Skip_Thoughts(AbstractEncoder):
    def __init__(self):
        st_path = '/home/mehre/[DRIVE]/[Dev]/[AI]/skip-thoughts-master/'
        skipthoughts.path_to_models = st_path
        skipthoughts.path_to_tables = st_path
        skipthoughts.path_to_umodel = st_path + 'uni_skip.npz'
        skipthoughts.path_to_bmodel = st_path + 'bi_skip.npz'
        model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(model)

    def use(self, sentences_to_embed):
        definitions_emb = list()
        for definition in sentences_to_embed:
            definitions_emb.append(self.encoder.encode([definition])[0])

        return definitions_emb
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

def experiment1(encoder):
    train_x, train_y, test_x, test_y = utils.load_dataset('data/train.csv', 'data/trial.csv', 'data/test.labels', partition=None)
    train_x = numpy.pad(train_x, (1, 0), 'constant', constant_values=0)
    train_x[0] = u"angry , sad , joy , disgust , fear , surprise"


    train_x_list = list()

    for sentence in train_x:
        train_x_list.append(unicode(sentence))

    print( train_x_list[0:10])

    encoder_instance = encoder(train_x_list)






if __name__ == '__main__':
    experiment1(Infersent)




