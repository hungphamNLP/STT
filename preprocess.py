import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse


rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 


with open('s.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# lặp qua intent để lưu tag and input training
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence

        # add to our words list
        all_words.extend(pattern)
        # add to xy pair
        xy.append((pattern, tag))
        
x=[]
y=[]
for i in xy :
  x.append(i[0])
  y.append(i[1])



df = pd.DataFrame(list(zip(x, y)),
               columns =['text', 'label'])

#prepare data
X_tokenize=[]
for sample in x:
  text = rdrsegmenter.tokenize(sample)
  text = ' '.join([' '.join(x) for x in text])
  X_tokenize.append(text)