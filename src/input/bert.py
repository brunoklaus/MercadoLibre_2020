'''
Created on 13 de nov de 2020

@author: klaus
'''

from sentence_transformers import SentenceTransformer
train_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
from scipy  import spatial
import numpy as np
def cos_sim(x,y):
    return 1  - spatial.distance.cosine(x,y)

def bert():
    print('SMARTWATCH XIAOMI MI'.capitalize())
    
    sentences = ['celular xiaomi',
                 'celular xiaomi redmi 7',
                 
                 'Celular Xiaomi Redmi Note 7 - 128gb - Fone Capa Pelicula 12x'.lower(),
                 'moder da josi']
    sentence_embeddings = train_model.encode(sentences)
    print(sentence_embeddings.shape)
    
    print(spatial.distance.cdist(sentence_embeddings,sentence_embeddings, 'euclidean'))


if __name__ == "__main__":
    bert()