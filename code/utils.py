import numpy as np
from random import randint
import os
import json
import pickle
import nltk.data
from pyvi import ViTokenizer
from sklearn.svm import LinearSVC
from gensim import corpora, matutils
from sklearn.metrics import classification_report
from extract_data import *

SPECIAL_CHARACTER = '01234567890123456789%@$.,=+-!;/()*"&^:#|\n\t\''
DICTIONARY_PATH = 'dictionary.txt'
STOPWORD_PATH = 'stopwords-nlp-vi.txt'

class DataLoader(object):

    def __init__(self, labels, contents):
        self.labels = labels
        self.contents = contents

    def get_json_format(self):
        data = []
        assert len(self.labels) == len(self.contents)
        for i in range(len(self.labels)):
            data.append({
                'label' : self.labels[i],
                'content' : self.contents[i]
            })
        return data

class FileReader(object):
    def __init__(self, filePath, encoder=None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-16le'
    
    def read(self): 
        with open(self.filePath) as f:
            s = f.read()
        return s
    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords
    
    def content(self):
        s = self.read()
        return s.decode(self.encoder)
    
    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)

class FileStore(object):
    
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        dictionary.save_as_text(self.filePath)

    def save_pickle(self,  obj):
        outfile = open(self.filePath, 'wb')
        fastPickler = pickle.Pickler(outfile, pickle.HIGHEST_PROTOCOL)
        fastPickler.fast = 1
        fastPickler.dump(obj)
        outfile.close()

class NLP(object):
    def __init__(self, text=None):
        self.text = text
        self.__set_stopword()

    def __set_stopword(self):
        self.stopwords = FileReader(STOPWORD_PATH).read_stopwords()
    
    def segmentation(self):
        return ViTokenizer.tokenize(self.text)
    
    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(SPECIAL_CHARACTER).lower() for x in text.split()]
        except TypeError:
            return []
    
    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8-sig') not in self.stopwords]

class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data
    
    def __build_dictionary(self):
        print ("Building dictionary...")
        dict_words = []
        i = 0
        for text in self.data:
            i+=1
            print("Build dictionary... Step {}/{}".format(i, len(self.data)))
            words = NLP(text = text['content']).get_words_feature()
            dict_words.append(words)
        FileStore(filePath=DICTIONARY_PATH).store_dictionary(dict_words)
    
    def __load_dictionary(self):
        if (os.path.exists(DICTIONARY_PATH) == False):
            self.__build_dictionary()
        self.dictionary = FileReader(DICTIONARY_PATH).load_dictionary()
    
    def __build_dataset(self):
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print("Build dataset... Step {}/{}".format(i, len(self.data)))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['label'])
        
    def get_dense(self, text):
        self.__load_dictionary()
        words = NLP(text).get_words_feature()
        # Bag of words 
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense
    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels
