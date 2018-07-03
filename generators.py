import numpy
import keras
import utils
import config

class MySMGenerator(keras.utils.Sequence):
    def __init__(self, filepath, batch, embedding_size, validation_split=0.1, batch_change=None):
        self.file = open(filepath, 'r', newline='')
        self.line_count = 0
        with open(filepath) as f:
            for l in f:
                self.line_count += 1
        self.line_count = numpy.floor(self.line_count*(1-validation_split))
        self.batch_change = batch_change
        self.filepath = filepath
        self.batch_size = batch
        self.embedding_size = embedding_size
        self.n_epochs = 0
        print("...Train generator initialized:")
        print("......lines: {}".format(self.line_count))
        print("......batch size: {}".format(self.batch_size))
        print("......embedding size: {}".format(self.embedding_size))
        print("......number of batches: {}".format(self.__len__()))
    def on_epoch_end(self):
        self.n_epochs += 1
        if self.batch_change is not None:
            if self.n_epochs in self.batch_change:
                self.batch_size *= 2
        self.file.close()
        self.file = open(self.filepath, 'r', newline='')
    def __len__(self):
        return int(numpy.floor(self.line_count / self.batch_size))
    def __getitem__(self, index):
        result_x = numpy.zeros((self.batch_size, self.embedding_size))
        result_y = numpy.zeros((self.batch_size, 6))
        for i in range(0, self.batch_size):
            line = self.file.readline()
            result_x[i] = numpy.fromstring(line.split(',')[1][1:-1], dtype=numpy.float32, sep=' ')
            label = line.split(',')[0]
            result_y[i] = utils.labels_to_indices(numpy.asarray([label]), config.labels_to_index, 6)
        return result_x, result_y

class MySMValidationGenerator(keras.utils.Sequence):
    def __init__(self, filepath, batch, embedding_size, validation_split=0.1):
        self.file = open(filepath, 'r', newline='')
        self.total_line_count = 0
        with open(filepath) as f:
            for l in f:
                self.total_line_count += 1
        self.validation_split = validation_split
        self.train_line_count = int(numpy.floor(self.total_line_count*(1-self.validation_split)))
        self.val_line_count = int(numpy.ceil(self.total_line_count*self.validation_split))
        for i in range(0, self.train_line_count):
            line = self.file.readline()
        self.filepath = filepath
        self.batch_size = batch
        self.embedding_size = embedding_size
        print("...Validation generator initialized:")
        print("......lines: {}".format(self.val_line_count))
        print("......batch size: {}".format(self.batch_size))
        print("......embedding size: {}".format(self.embedding_size))
        print("......number of batches: {}".format(self.__len__()))
    def __len__(self):
        return int(numpy.floor(self.val_line_count / self.batch_size))
    def __getitem__(self, index):
        result_x = numpy.zeros((self.batch_size, self.embedding_size))
        result_y = numpy.zeros((self.batch_size, 6))
        for i in range(0, self.batch_size):
            line = self.file.readline()
            if line == '':
                self.file.close()
                self.file = open(self.filepath, 'r', newline='')
                for ll in range(0, self.train_line_count):
                    line = self.file.readline()
                line = self.file.readline()
            result_x[i] = numpy.fromstring(line.split(',')[1][1:-1], dtype=numpy.float32, sep=' ')
            label = line.split(',')[0]
            result_y[i] = utils.labels_to_indices(numpy.asarray([label]), config.labels_to_index, 6)[0]
        return result_x, result_y


class MyGenerator(keras.utils.Sequence):
    def __init__(self, filepath, batch, embedding_size, validation_split=0.1, batch_change=None):
        self.file = open(filepath, 'r', newline='')
        self.line_count = 0
        with open(filepath) as f:
            for l in f:
                self.line_count += 1
        self.line_count = numpy.floor(self.line_count*(1-validation_split))
        self.batch_change = batch_change
        self.filepath = filepath
        self.batch_size = batch
        self.embedding_size = embedding_size
        self.n_epochs = 0
        print("...Train generator initialized:")
        print("......lines: {}".format(self.line_count))
        print("......batch size: {}".format(self.batch_size))
        print("......embedding size: {}".format(self.embedding_size))
        print("......number of batches: {}".format(self.__len__()))
    def on_epoch_end(self):
        self.n_epochs += 1
        if self.batch_change is not None:
            if self.n_epochs in self.batch_change:
                self.batch_size *= 2
        self.file.close()
        self.file = open(self.filepath, 'r', newline='')
    def __len__(self):
        return int(numpy.floor(self.line_count / self.batch_size))
    def __getitem__(self, index):
        result_x = numpy.zeros((self.batch_size, self.embedding_size))
        result_y = numpy.zeros((self.batch_size, 6))
        for i in range(0, self.batch_size):
            line = self.file.readline()
            result_x[i] = numpy.fromstring(line.split(',')[1][1:-1], dtype=numpy.float32, sep=' ')
            label = line.split(',')[0]
            result_y[i] = utils.labels_to_indices(numpy.asarray([label]), config.labels_to_index, 6)
        return result_x, result_y