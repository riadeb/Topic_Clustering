#!/usr/bin/env python
# coding: utf-8
# Built by Riade Benbaki and Haolin Pan
# Topic Clustering Details

# set up
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import metrics

from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

class Topic_clustering(object):

    def __init__(self):
        self.data = requests.get(
            "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-1-2019/self-dialogs.json").json()
        self.numofclasses = 7

        # data set loading
        self.X = None
        self.y = None
        self.samplestotakefromeachclass = 1400
        self.classes_arr = ['auto', 'coffee', 'movie', 'non-opening', 'pizza', 'restaurant', 'uber']
        use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.use = hub.load(use_url)
        print("module use loaded")
        self.elmo = hub.load("https://tfhub.dev/google/elmo/3")
        print("module elmo loaded")
        self.nnlm = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
        print("module nnlm loaded")
        self.gnews = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")
        print("Gnews Swivel loaded")

        # data zoo to store all the data already embedded
        self.data_zoo = {}
        self.embedding_models = []

        # hyper_parameter candidates:
        self.loss = ['categorical_crossentropy', 'cosine_similarity', 'binary_crossentropy',
                     'kullback_leibler_divergence','hinge']
        self.optimizer = ['adam', 'sgd']

        # restore the optimal hyper parameter for each models, the initial value is the result of our experience
        # If you want to make experience yourself, just run hyper_parameter_tuning
        self.best_hyper_parameter = {'elmo': ('cosine_similarity', 'adam', 0.915502755778993),
                                     'use': ('cosine_similarity', 'adam', 0.8861844411292173),
                                     'nnlm': ('kullback_leibler_divergence', 'adam', 0.7370132560362909),
                                     'gnew': ('kullback_leibler_divergence', 'adam', 0.4088656873174067)}

        # early stop epochs:
        self.best_early_stop = {'elmo': 47, 'use': 83, 'nnlm': 176, 'gnew': 176}

        # model zoo
        self.model_zoo = {}

        for em in ['elmo', 'use']:
            self.model_zoo[em] = tf.keras.Sequential()
            if em == 'elmo':
                self.model_zoo[em].add(tf.keras.layers.InputLayer(input_shape=[1024, ]))
            else:
                self.model_zoo[em].add(tf.keras.layers.InputLayer(input_shape=[512, ]))
            self.model_zoo[em].add(tf.keras.layers.Dense(self.numofclasses, activation='softmax'))
            self.model_zoo[em].compile(loss=self.best_hyper_parameter[em][0], optimizer=self.best_hyper_parameter[em][1],
                         metrics=[metrics.mae, metrics.categorical_accuracy])
            if os.path.exists("topic_clustering_model/%s.h5"%em):
                self.model_zoo[em].load_weights("topic_clustering_model/%s.h5"%em)

    # Word Embedding Pretrained Models:
    def NNLM(self, x):
        return self.nnlm.signatures['default'](tf.convert_to_tensor(x))

    def ELMO(self, x):
        return self.elmo.signatures['default'](tf.convert_to_tensor(x))

    def USE(self, x):
        return self.use(x)

    def GNEW(self, x):
        return self.gnews(x)

    def transform_array(self, Y):  # transforms Y from class name array to appropriate format
        classes_arr = np.unique(Y)
        classes_dict = dict()
        for i, class_n in enumerate(np.unique(Y)):
            classes_dict[class_n] = i
        for i in range(len(Y)):
            toadd = [0] * self.numofclasses
            toadd[classes_dict[Y[i]]] = 1
            Y[i] = list(toadd)
        return np.array(Y)

    def my_train_test_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def data_precessing(self):
        X, Y = [], []
        dd = {'auto': 0,
              'coffee': 0,
              'movie': 0,
              'pizza': 0,
              'restaurant': 0,
              'uber': 0,
              'non-opening': 0}

        for ut in self.data:
            sent = ut["utterances"][0]
            class_to_add = ut["instruction_id"].split("-")[0]
            if dd[class_to_add] < self.samplestotakefromeachclass:
                X.append(sent["text"])
                Y.append(class_to_add)
                dd[class_to_add] += 1

            class_to_add = 'non-opening'
            if dd[class_to_add] < self.samplestotakefromeachclass:
                X.append(ut["utterances"][np.random.randint(1, len(ut["utterances"]))]["text"])
                Y.append(class_to_add)
                dd[class_to_add] += 1

        print("The numbers of samples we take from the original data: ")
        for k in dd.keys():
            print(k, ": \t", dd[k])
        print("Total:\t ", len(X))
        self.X = np.array(X)
        self.y = self.transform_array(Y)
        return

    def word_embedding(self):
        """Embed the data and store them in the data zoo"""
        """If already embedded, load them from csv"""

        # If data is not embedded, embed them
        if not os.path.exists("data_USE.csv"):
            X_use = np.array(self.USE(self.X))
            np.savetxt("data_USE.csv", X_use, delimiter=",")

        if not os.path.exists("data_GNEW.csv"):
            X_gnew = np.array(self.gnews(self.X))
            np.savetxt("data_GNEW.csv", X_gnew, delimiter=",")

        if not os.path.exists("data_ELMo.csv"):
            X_elmo = []
            for i in range(len(self.X)):
                X_elmo.append(np.array(self.ELMO([self.X[i]])['default']))
            X_elmo = np.array(X_elmo)
            X_elmo = np.reshape(X_elmo, [X_elmo.shape[0], X_elmo.shape[2]])
            np.savetxt("data_ELMo.csv", X_elmo, delimiter=",")

        if not os.path.exists("data_NNLM.csv"):
            X_nnlm = np.array(self.NNLM(self.X)['default'])
            np.savetxt("data_NNLM.csv", X_nnlm, delimiter=",")

        if not os.path.exists("data_y.csv"):
            np.savetxt("data_y.csv", self.y, delimiter=",")

        # Get data from csv
        X_elmo = np.genfromtxt('data_ELMo.csv', delimiter=',')
        X_use = np.genfromtxt('data_USE.csv', delimiter=',')
        X_nnlm = np.genfromtxt('data_NNLM.csv', delimiter=',')
        X_gnew = np.genfromtxt('data_GNEW.csv', delimiter=',')
        #y = np.genfromtxt('data_y.csv', delimiter=',')
        y = self.y;
        print("training data loaded")

        # Split the data into development set and test set
        # the test set is not used until the final evaluation
        X_elmo, X_test_elmo, y_elmo, y_test_elmo = self.my_train_test_split(X_elmo, y, 0.1)
        X_use, X_test_use, y_use, y_test_use = self.my_train_test_split(X_use, y, 0.1)
        X_nnlm, X_test_nnlm, y_nnlm, y_test_nnlm = self.my_train_test_split(X_nnlm, y, 0.1)
        X_gnew, X_test_gnew, y_gnew, y_test_gnew = self.my_train_test_split(X_gnew, y, 0.1)

        self.data_zoo['elmo'] = {'dev': (X_elmo, y_elmo), 'test': (X_test_elmo, y_test_elmo)}
        self.data_zoo['use'] = {'dev': (X_use, y_use), 'test': (X_test_use, y_test_use)}
        self.data_zoo['nnlm'] = {'dev': (X_nnlm, y_nnlm), 'test': (X_test_nnlm, y_test_nnlm)}
        self.data_zoo['gnew'] = {'dev': (X_gnew, y_gnew), 'test': (X_test_gnew, y_test_gnew)}
        self.embedding_models = list(self.data_zoo.keys()) # list of the name of data embedded

    def f1_score_model(self, model, X, y):
        """Get the F1 score of a specific model on a specific data (X, y)"""

        prediction = np.argmax(model.predict(X), axis=1)
        y = np.argmax(y, axis=1)
        f1_macro = f1_score(y, prediction, average='macro')
        f1_micro = f1_score(y, prediction, average='macro')
        print("f1_macro: ", f1_score(y, prediction, average='macro'))
        print("f1_micro: ", f1_score(y, prediction, average="micro"))
        print("f1_weighted: ", f1_score(y, prediction, average="weighted"))
        return f1_macro, f1_micro

    def test_model(self, X, y, epochs=3, optimizer='adam', loss='categorical_crossentropy'):
        """Test the performance of the model on a random split of validation set"""
        """return the f1 scores on the validation set"""

        X_train, X_dev, y_train, y_dev = self.my_train_test_split(X, y)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.numofclasses, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.mae, metrics.categorical_accuracy])
        model.fit(X_train, y_train, epochs=epochs, batch_size=64)
        return self.f1_score_model(model, X_dev, y_dev)

    def hyper_parameter_tuning(self):
        self.best_hyper_parameter = {}

        for em in self.data_zoo.keys(): # em for Embedding Model
            print("Tuning ", em)
            X, y = self.data_zoo[em]['dev']
            hyper_parameter_benchmark = {}
            best_f1 = 0

            X_train, X_dev, y_train, y_dev = self.my_train_test_split(X, y)

            for l in self.loss:
                for opt in self.optimizer:
                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.Dense(self.numofclasses, activation='softmax'))
                    model.compile(loss=l, optimizer=opt, metrics=[metrics.mae, metrics.categorical_accuracy])
                    model.fit(X_train, y_train, epochs=3, batch_size=64)
                    f1 = self.f1_score_model(model, X_dev, y_dev)

                    hyper_parameter_benchmark[(l, opt)] = f1
                    print(l, ", ", opt, ": ", hyper_parameter_benchmark[(l, opt)])
                    if hyper_parameter_benchmark[(l, opt)][0] > best_f1:
                        best_f1 = hyper_parameter_benchmark[(l, opt)][0]
                        self.best_hyper_parameter[em] = (l, opt, best_f1)
                    print(128 * "=")
                    print()

    def early_stop_searching(self):
        """Iterate through the number of epochs to find the optimal time for early stop"""

        early_stop = {}

        for em in self.embedding_models:
            print("Searching ", em)
            X, y = self.data_zoo[em]['dev']
            X_train, X_dev, y_train, y_dev = self.my_train_test_split(X, y)
            f1_seq = []
            epoch_seq = []
            epoch = 3;
            while (epoch < 200):
                epoch_seq.append(epoch)
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(self.numofclasses, activation='softmax'))
                model.compile(loss=self.best_hyper_parameter[em][0], optimizer=self.best_hyper_parameter[em][1],
                              metrics=[metrics.mae, metrics.categorical_accuracy])
                model.fit(X_train, y_train, epochs=epoch, batch_size=128)
                f1 = self.f1_score_model(model, X_dev, y_dev)
                f1_seq.append(f1[0])

                epoch = int(1.2 * epoch + 1)

            early_stop[em] = {'epoch': epoch_seq, 'f1': f1_seq}

        plt.plot(early_stop['elmo']['epoch'], early_stop['elmo']['f1'], early_stop['use']['epoch'],
                 early_stop['use']['f1'], early_stop['nnlm']['epoch'], early_stop['nnlm']['f1'],
                 early_stop['gnew']['epoch'], early_stop['gnew']['f1'])
        plt.show()

        plt.plot(early_stop['elmo']['epoch'], early_stop['elmo']['f1'], early_stop['use']['epoch'],
                 early_stop['use']['f1'])
        plt.show()

        self.best_early_stop = {}
        for em in self.embedding_models:
            self.best_early_stop[em] = early_stop[em]['epoch'][np.argmax(np.array(early_stop[em]['f1']))]
            print("The best epoch of", em, "is: ", early_stop[em]['epoch'][np.argmax(np.array(early_stop[em]['f1']))])


    def evaluation_test_set(self):
        """Evaluation on the test set"""
        """The performance justifies our final choice of model"""

        f1_test = {}

        for em in self.embedding_models:
            X_train, y_train = self.data_zoo[em]['dev']
            X_test, y_test = self.data_zoo[em]['test']
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(self.numofclasses, activation='softmax'))
            model.compile(loss=self.best_hyper_parameter[em][0], optimizer=self.best_hyper_parameter[em][1],
                          metrics=[metrics.mae, metrics.categorical_accuracy])
            model.fit(X_train, y_train, epochs=self.best_early_stop[em], batch_size=128)
            model.save_weights("topic_clustering_model/%s.h5" % em)
            f1_test[em] = self.f1_score_model(model, X_test, y_test)
        return f1_test

    def prediction(self, em, phrs):
        if em == "use":
            X = np.array(self.USE(phrs))
            return ([self.classes_arr[i] for i in np.argmax(self.model_zoo['use'].predict(X), axis=1)])
        X = []
        for i in range(len(phrs)):
            X.append(np.array(self.ELMO([phrs[i]])['default']))
        X = np.array(X)
        X = np.reshape(X, [X.shape[0], X.shape[2]])
        return ( [ self.classes_arr[i] for i in np.argmax(self.model_zoo['elmo'].predict(X), axis=1)])


if __name__ == "__main__":
    TCM = Topic_clustering()
    TCM.data_precessing()
    TCM.word_embedding()
    # TCM.hyper_parameter_tuning() optional
    # TCM.early_stop_searching() optional
    f1_test = TCM.evaluation_test_set()

    for em in TCM.embedding_models:
        print("The f1 score of ", em, " is ", f1_test[em][0])


    test_phrases = ["Where is the nearest Starbucks ?", "i need to repair my car",
                    "I need a ride from home", "I want to order something to eat", "can you activate",
                    "I want a table in center city", "Ok that's it!"]
    res = TCM.prediction('use', test_phrases)
    print("Universal Standard Embedding: ")
    for i in range(len(res)):
        print("The predicted topic of \"{} \" is : {}".format(test_phrases[i], res[i]))
    res = TCM.prediction('elmo', test_phrases)
    print(128 * "=")
    print()

    print("ELMo")
    for i in range(len(res)):
        print("The predicted topic of \"{} \" is : {}".format(test_phrases[i], res[i]))
