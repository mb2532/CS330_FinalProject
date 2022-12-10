# Code to create BERT sentence embeddings from Google Reviews 
# To be used as graph edge attributes 

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import networkx as nx

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

from sentence_transformers import SentenceTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

columns_name = ['place_index', 'user_index', 'rating', 'review_text']

city_files = ["sanantonio_train.tsv"]

def create_review_embedding(city_file):
    filepath = "data/CITIES/"
    review_df = pd.read_csv(filepath + city_file, sep="\t")[columns_name]
    sentences = list(review_df['review_text'])
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    for i, sentence in enumerate(sentences):
        if type(sentence) != type(''): 
            sentences[i] = ''
    sentence_embeddings = sbert_model.encode(sentences)
    np.savetxt(city_file.split('-')[0] + '_emb.txt', sentence_embeddings)

for city_file in city_files:
    print(city_file)
    create_review_embedding(city_file)