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
from sklearn.model_selection import train_test_split
from utils import *
from extract_data import *

class Classifier(object):
    def __init__(self, features_train = None, labels_train = None, 
                features_test = None, labels_test = None, estimator=LinearSVC(random_state=0)):
        self.features_train = features_train
        self.labels_train = labels_train
        self.features_test = features_test
        self.labels_test = labels_test
        self.estimator = estimator
    
    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        self.__training_result()
    
    def save_model(self, filePath):
        FileStore(filePath=filePath).save_pickle(obj=est)

    def __training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        print(classification_report(y_true, y_pred))


if __name__=="__main__":
    
    model_name = "linearSVC_model.pkl"
    traindata_file = 'traindatatopic.txt'
    raw_data = read_all_file(traindata_file)
    raw_document = get_document_row_data(raw_data)
    raw_labels, raw_contents = extract_data(raw_document)
    raw_contents_train, raw_contents_test,raw_labels_train, raw_labels_test = train_test_split(raw_contents, raw_labels, test_size=0.2, random_state=42)
    train_data = DataLoader(raw_labels_train, raw_contents_train).get_json_format()
    test_data = DataLoader(raw_labels_test, raw_contents_test).get_json_format()
    features_train, labels_train = FeatureExtraction(data=train_data).get_data_and_label()
    features_test, labels_test = FeatureExtraction(data=test_data).get_data_and_label()
    est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train,
            labels_test=labels_test)
    est.training()
    est.save_model(model_name)
