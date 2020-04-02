#!/usr/bin/env python
# coding: utf-8


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

import json



class Intent_classifiction(object):
    def __init__(self):
        self.url_data = "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-1-2019/self-dialogs.json"
        self.data = requests.get(self.url_data).json()
        
        # Phrases to intents
        self.phrs2intents = {}
        
        # Phrases embedded
        self.phrs2vec = {}
        
        # phrases lists
        self.X = None
        
        # Word Embedding models loading
        use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.use = hub.load(use_url)
        print("module use loaded")
        self.elmo = hub.load("https://tfhub.dev/google/elmo/3")
        print("module elmo loaded")

    def ELMO(self, x):
        return self.elmo.signatures['default'](tf.convert_to_tensor(x))

    def USE(self, x):
        return self.use(x)

    def intents_clustering(self):
        """ Establishing a dictionary from phrases to intents """
        self.phrs2intents = {}
        number_of_other = 10000;
        for i in range(len(self.data)):
            for ut in self.data[i]['utterances']:
                if ut['speaker'] == 'USER':
                    if 'segments' in ut.keys():
                        for seg in ut['segments']:
                            if 'annotations' in seg.keys():
                                for anno in seg['annotations']:
                                    name = anno['name']
                                    if ut['text'] not in self.phrs2intents.keys():
                                        self.phrs2intents[ ut['text'] ] = [name]
                                    elif name not in self.phrs2intents[ ut['text'] ]:
                                        self.phrs2intents[ ut['text'] ].append(name)
                    else:
                        if number_of_other > 0:
                            self.phrs2intents[ ut['text'] ] = ['other']
                            number_of_other -= 1
        self.X = np.array(list(self.phrs2intents.keys()))
        
    def word_embedding(self):
        self.intents_clustering()
        print("intents clustering completed!")
        if not os.path.exists("phr_embedded.json"):
            self.phrs2vec = {}
            
            for i in range(len(self.X)):
                if i % 1000 == 0:
                    print("Embedding Progress: ", i / len(self.X))
                self.phrs2vec[ self.X[i] ]  = str( list( np.array(self.ELMO([self.X[i]])['default'][0]) ))
            with open('phr_embedded.json', 'w') as fp:
                json.dump(self.phrs2vec, fp)
                           
        with open('phr_embedded.json', 'r') as fp:
            self.phrs2vec = json.load(fp)
        print("embedded data loading completed")              
        
        self.phrs = list(self.phrs2vec.keys())
        for i in range( len(self.phrs) ):
            if i % 1000 == 0:
                    print("Embedding Progress: ", i / len(self.X))
            self.phrs2vec[self.phrs[i]] = np.array( eval( self.phrs2vec[self.phrs[i]]) )
    
    def get_embedded_data_and_clustered_data(self):
        return self.phrs2intents, self.phrs2vec



# # Sub Intents Study
# ------


class Intent_detection(object):
    ''' In this class , we use the notations below'''
    ''' topic means: the grand theme of the conversation, e.g.: "restaurant" '''
    ''' subtopic means: the small theme belonging to the topic, e.g. "restaurant.time" '''
    ''' intents means: the intent of the user, which kind of express the attitude of user, e.g. "restaurant.time.accept" '''

    def __init__(self, p2i, p2v):
        self.p2i = p2i
        self.p2v = p2v
        self.topic = ['auto', 'coffee', 'movie', 'pizza', 'restaurant', 'uber', 'other']
        self.phrs_zoo = np.array(list(p2i.keys()))
        self.topic2phrs_zoo = {'auto':[], 'coffee':[], 'movie':[], 'pizza':[], 'restaurant':[], 'uber':[], 'other':[]}
        
        # Load word embedding model
        self.elmo = hub.load("https://tfhub.dev/google/elmo/3")
        print("module elmo loaded")
        
        # Load the phrases zoo
        for t in self.topic:
            for i in range(len(self.phrs_zoo)):
                if self.p2i[ self.phrs_zoo[i] ][0].find(t) > -1:
                    self.topic2phrs_zoo[ t ].append(self.phrs_zoo[i])
     
        #  the sub_topic dictionary stores the subtopics of each topic and their coding
        self.topic2sub_topic = {'restaurant': {"time": 1, "location": 2, "num": 3, "name": 4, "type": 5},
                          'auto': {"name": 1, "year": 2, "reason": 3, "date": 4},
                          'movie': {"time": 1, "location": 2, "num": 3, "name": 4, "type": 5, "price": 6},
                          'coffee': {"size": 1, "location": 2, "num": 3, "name": 4, "type": 5, "preference": 6},
                          'pizza': {"size": 1, "location": 2, "preference": 3, "name": 4, "type": 5},
                          'uber': {"time": 1, "location": 2, "num": 3, "duration": 4, "type": 5},
                          'other':{"other":1}  }
        
        # load phrases to each subtopic
        self.phrs2sub_topic = {'auto':{}, 'coffee':{}, 'movie':{}, 'pizza':{}, 'restaurant':{}, 'uber':{},  'other':{} }
        
        for t in self.topic:
            if t != 'other':
                sub_topic = list( self.topic2sub_topic[t].keys() )
                for p in self.topic2phrs_zoo[t]:
                    sub_topic_of_p =  len(sub_topic) * [ 0 ]
                    for i in self.p2i[p]:
                        for st in sub_topic:
                            if i.find(st) > -1:
                                sub_topic_of_p[ self.topic2sub_topic[t][st] - 1 ] = 1
                    self.phrs2sub_topic[t][p] = tuple(sub_topic_of_p)
        
        
        #       Two dictionarys to stock the training set according to topics
        self.model_zoo = {'auto':{}, 'coffee':{}, 'movie':{}, 'pizza':{}, 'restaurant':{}, 'uber':{}};
        
    
    def my_train_test_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def f1_score_model(self, model, X, y):
        """Get the F1 score of a specific model on a specific data (X, y)"""

        prediction = model.predict_classes(X)
        f1_macro = f1_score(y, prediction, average='macro')
        f1_micro = f1_score(y, prediction, average='macro')
        print("f1_macro: ", f1_score(y, prediction, average='macro'))
        print("f1_micro: ", f1_score(y, prediction, average="micro"))
        print("f1_weighted: ", f1_score(y, prediction, average="weighted"))
        return f1_macro, f1_micro
    
    def get_data(self, t, st, total_data_number = 10000):
        """ Get the data for training and validation for specific """
        
        X = []
        y = []
        num_pos = 0
        num_neg = 0
        num_other = 0
        for p in self.topic2phrs_zoo[t]:
            if num_pos > total_data_number // 2 and num_neg > total_data_number // 4:
                break
            if self.phrs2sub_topic[t][p][  self.topic2sub_topic[t][st] - 1  ] > 0:
                if num_pos >= -0:
                    X.append( p2v[p] )
                    y.append( 1 )
                    num_pos += 1
            else:
                if num_neg <= num_pos // 2:
                    X.append( p2v[p] )
                    y.append( 0 )
                    num_neg += 1
        labeled_sample = len(y)
        for i in range(len(y) // 3):
            p = self.topic2phrs_zoo['other'][np.random.randint(len( ID.topic2phrs_zoo['other'] ))]
            X.append(p2v[p])
            y.append(0)
            num_other += 1;
        print("We get: ")
        print(num_pos, " postive sample")
        print(num_neg, " sample of wrong intents")
        print(num_other, " sample of no intents")
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """ Training all the subtopic and topics"""
        self.best_epoch = {'auto':{}, 'coffee':{}, 'movie':{}, 'pizza':{}, 'restaurant':{}, 'uber':{} }
        self.best_f1 = {'auto':{}, 'coffee':{}, 'movie':{}, 'pizza':{}, 'restaurant':{}, 'uber':{} }
        for t in self.topic:
            if t != 'other':
                for st in self.topic2sub_topic[t].keys():

                    print("Now training the classsfier for topic: ", t, " ; intent: ", st)
                    print(128 * "=")
                    print("Input: str; Output: boolean(if the str contents the intent: ", st, " ).")
                    print(64 * "-")
                    X, y = self.get_data(t, st)
                    print("data_loaded!")
                    X_train, X_dev, y_train, y_dev = self.my_train_test_split(X, y)
                    best_f1 = 0
                    for e in range(1,10):
                        model = tf.keras.Sequential()
                        model.add(tf.keras.layers.InputLayer(input_shape=[1024, ]))
                        model.add(tf.keras.layers.Dense(64, activation='relu'))
                        model.add(tf.keras.layers.Dense(64, activation='relu'))
                        model.add(tf.keras.layers.Dense(1, activation='relu'))
                        model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=[metrics.mae, metrics.categorical_accuracy])
                        model.fit(X_train, y_train, epochs=e, batch_size=128)
                        print("f1_score on dev set: ")
                        f1 = self.f1_score_model(model, X_dev, y_dev)[0]
                        if f1 > best_f1:
                            self.model_zoo[t][st] = model
                            model.save_weights("intent_detection_model/%s/%s.h5" %(t,st))
                            self.best_epoch[t][st] = e
                            self.best_f1[t][st] = f1
                            best_f1 = f1

                        print(64*"=")
                    print()
    
    def load_model(self):
        """Load all the models, if model does not exist, retrain it and store it"""
        for t in self.topic:
            if t != "other":
                print("Loading models of topic: ", t)
                for st in self.topic2sub_topic[t].keys():
                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.InputLayer(input_shape=[1024, ]))
                    model.add(tf.keras.layers.Dense(64, activation='relu'))
                    model.add(tf.keras.layers.Dense(64, activation='relu'))
                    model.add(tf.keras.layers.Dense(1, activation='relu'))
                    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=[metrics.mae, metrics.categorical_accuracy])

                    if not os.path.exists("intent_detection_model/%s/%s.h5" %(t,st)):
                        print("Now training the classsfier for topic: ", t, " ; intent: ", st)
                        print(64 * "=")
                        X, y = self.get_data(t, st)
                        print("data_loaded!")
                        X_train, X_dev, y_train, y_dev = self.my_train_test_split(X, y)
                        model.fit(X_train, y_train, epochs=3, batch_size=128)
                        model.save_weights("intent_detection_model/%s/%s.h5" %(t,st))
                        print("f1_score on dev set: ")
                        self.f1_score_model(model, X_dev, y_dev)
                        print(64*"=")
                        print()
                    else:
                        model.load_weights("intent_detection_model/%s/%s.h5" %(t,st))
                    self.model_zoo[t][st] = model
                
    def ELMO(self, x):
        return self.elmo.signatures['default'](tf.convert_to_tensor(x))
    
    def predict(self, phr, t):
        '''Enter one phrase and get the prediction of if this phrase contains each intent'''
        '''phr: str, t: str'''
        
        X = np.array(self.ELMO([phr]) ['default'])
            
        st_array = list( self.topic2sub_topic[t].keys() )
        res = []
        for st in st_array:
            res.append( self.model_zoo[t][st].predict(X)[0][0] )
        
        return res




if __name__ == "__main__":
    IC = Intent_classifiction()
    IC.word_embedding()
    p2i, p2v = IC.get_embedded_data_and_clustered_data()
    ID = Intent_detection(p2i, p2v)
    # ID.train_model()
    ID.load_model()
    test_phrases = np.array([["Get me a restaurant not far from here!"], ["This Evening at 8"], ["I want this"],
                    ["We have totally 4"], ["My dad, mom, me and my sister"], ["Do you have some recommends"],
                    ["Thank you! Bye"]])

    t = 'restaurant'
    for phr in test_phrases:
        intent = []
        res = ID.predict( phr[0] , t  )
        sub_topic_array = list( ID.topic2sub_topic[t].keys() )
        print(phr[0])
        for st in sub_topic_array:
            if res[ ID.topic2sub_topic[t][st] - 1] > 0.5:
                intent.append(st)
        
        if len(intent) > 0:
            print("Detected intents: ", end="")
            for i in intent:
                print(i, end=" ")
        else:
            print("No intent detected")
        print()
        print(32 * "-=")
        print()

