# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:50:40 2024

@author: Yiyang Liu

"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
from PIL import Image
import os, sys
from wordcloud import WordCloud, STOPWORDS
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
#import multiprocessing
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA


class Text_W2V():
    
    """
    Using Gensim W2V as the text encoder. The class contains the tokenizer for 
    this specific encoder, a zero-shot encoder, and a pretrained encoder.
    """
    
    def __init__(self, pretrained = False, pretrained_model = None, sent=None, **kwargs):
        
        """
        
        """
        mod = Word2Vec(sentences = sent, **kwargs)
        words = list(mod.wv.key_to_index.keys())
        if pretrained:
            Pre_Vector = api.load(pretrained_model)
            self.model = Word2Vec(**kwargs)
            self.model.build_vocab(sent)
            self.model.build_vocab([list(Pre_Vector.index_to_key)], update=True) 
            self.model.wv.vectors = Pre_Vector.vectors
            embedding = [ self.model.wv[k] if k in self.model.wv else np.zeros(kwargs['vector_size']) for k in words ]
            new_keyed_vectors = KeyedVectors(vector_size=kwargs['vector_size'])
            new_keyed_vectors.add_vectors(words, embedding)
            self.wv = new_keyed_vectors
            
        else:
            self.wv = mod.wv
            self.model = mod
            #self.embedding = list(self.model.wv.vectors)
            
    def get_keyed_vector(self):
        """
        

        Returns
        -------
        w2v keyed vector
        
        """
        
        return self.wv
    
    
    def save_model(self, filepath = 'text_w2v'):
        """
        

        Parameters
        ----------
        filepath : string, optional
            DESCRIPTION. The default is 'text_w2v'.

        Returns
        -------
        None.

        """
        
        self.model.save(filepath)
        
        
    def tokenizer(self, cleaned_text_file):
        
        
        text_input = pd.read_csv(cleaned_text_file)
        text_input = list(text_input['text_corrected'])



        text_input = [word_tokenize(s) for s in text_input]
        return text_input
        
    
        
            
    
    
class Visualizer():
    """
    This class is used to visualize the word-embedding.
    """
    
    def __init__(self, Keyed_Vectors, idx):
        """
        Parameters
        ----------
        key [list]
            DESCRIPTION.
        embedding [list]
            DESCRIPTION.
        idx [list]
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.words = list(Keyed_Vectors.index_to_key)
        self.vectors = list(Keyed_Vectors.vectors)
        self.idx = idx
    
        
    def show(self):
        """show the image! """

        TD = PCA(n_components = 2)
        x1,y1 = TD.fit_transform(self.vectors).T
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x1[self.idx],y1[self.idx], c='k',s = 3)

        for i in self.idx:
            ax.annotate(
                self.words[i],
                (x1[i], y1[i]),
                xytext=(2, 2),
                textcoords='offset points',fontsize = 6
                )

        plt.title('word embeddings')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()