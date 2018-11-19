import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import re 
import csv

document_pattern = r'<document>(.*?)</document>'
label_pattern = r'<label>(.*?)</label>'
content_pattern = r'<content>(.*?)</content>'

def read_all_file(filename):
    with open(filename, "r") as f:
        raw_data = f.read().replace("\n", " ")
    return raw_data

def get_document_row_data(raw_data):
    document_raw_data = re.findall(document_pattern, raw_data, flags=0)
    return document_raw_data

def extract_data(document_raw_data):
    labels = []
    contents = []
    for i in range(len(document_raw_data)):
        label = re.findall(label_pattern, document_raw_data[i], flags=0)
        if (label != [] and label[0] == 'oreign language'):
            labels.append('foreign language')
        elif (label == []):
            labels.append("other topics")
        else: 
            labels.append(label[0])
        content = re.findall(content_pattern, document_raw_data[i], flags=0)
        if (content != []):
            contents.append(content[0])
        else: 
            contents.append("None content")
    return labels, contents

def save_extracted_data_into_csv(csv_filename, labels, contents):
    with open(csv_filename, 'w', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(zip(labels, contents))
    