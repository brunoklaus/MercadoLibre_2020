'''
Created on 18 de nov de 2020

@author: klaus
'''


'''
Created on 16 de nov de 2020

@author: klaus
'''
import jsonlines
from folders import DATA_DIR, SUBMISSIONS_DIR
from input.read_input import TRAIN_LINES, NUM_DOMS, get_emb, get_emb_Kstr
import os
from os import path
import pandas as pd
import numpy as np
import igraph as ig
from input.read_input import read_item_data, get_mappings

from sentence_transformers import SentenceTransformer
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')



NUM_WORDS = 5

BS = 64

def preprocess_title(tit):
    tit = tit.lower().strip().split(' ')[:NUM_WORDS]
    tit = tit + [' ']*(NUM_WORDS-len(tit))
    return tit

def title_generator(mode='train'):
        df = read_item_data()
        NUM_DOMS = pd.unique(df['domain_id']).size
        dom_to_id = dict([(x,i) for i,x in enumerate(pd.unique(df['domain_id']))])
        NUM_DOMS = pd.unique(df['domain_id']).size
        
        BATCH_X = []
        BATCH_Y = []
        
        while True:
            line_id = 0
            for tit,dom in (zip(df['title'],df['domain_id'])):
                target = np.zeros((NUM_DOMS,),dtype=np.float32)
                target[dom_to_id[dom]] = 1
                
                tit = preprocess_title(tit)
                
                embeddings = sentence_model.encode(tit)
                
                BATCH_X.append(embeddings[None,:,:])
                BATCH_Y.append(target[None,:])
                
                if line_id % BS == 0:
                    X = np.concatenate(BATCH_X,axis=0)
                    Y = np.concatenate(BATCH_Y,axis=0)
                    BATCH_X = []
                    BATCH_Y = []
                    yield X,Y
                line_id += 1

@tf.function
def smooth_labels(labels, factor=0.001):
    # smooth the labels
    labels = tf.cast(labels,tf.float32)
    labels *= (1 - factor)
    labels += (factor / tf.cast(tf.shape(labels)[1],tf.float32))
    # returned the smoothed labels
    return labels

def xent(Y1,Y2):

    Y1 = tf.reshape(Y1,(-1,NUM_DOMS))
    Y2 = tf.reshape(Y2,(-1,NUM_DOMS))

    Y1 = smooth_labels(Y1)
    Y2 = smooth_labels(Y2)
    return - tf.reduce_sum(Y1 * tf.math.log(Y2),axis=-1)   

def get_model():
    from tensorflow import keras
    from tensorflow.keras import layers
    
    inp_x = keras.Input((NUM_WORDS,512))
    x = tf.keras.layers.BatchNormalization()(inp_x)
    x = tf.keras.layers.Dense(128,activation=tf.nn.relu)(inp_x)
    
    x = layers.LSTM(64)(x)          
    x = layers.Dense(NUM_DOMS)(x)
    x = layers.Softmax()(x)
                
    train_model = keras.Model(inputs=[inp_x],outputs=[x])
            
        


    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
    
    train_model.compile(optimizer=optimizer,loss=xent,metrics=['acc'])
    print(train_model.summary())
    return train_model


DOMAIN_IDENTIFIER_PATH = path.join(DATA_DIR,'model','domain_identifier.h5')

def train_model():
    train_model = get_model()
    from tensorflow import TensorShape as ts
    import tensorflow.keras as keras
    
    train_ds = tf.data.Dataset.from_generator( title_generator,output_types =(tf.float32,tf.float32),
                                               output_shapes =(ts([None,NUM_WORDS,512]),
                                                               ts([None,NUM_DOMS]))
                                               )
    train_model.load_weights(DOMAIN_IDENTIFIER_PATH)
    train_model.fit(
    x=train_ds,
    steps_per_epoch= TRAIN_LINES // BS,
    epochs=1)
    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    
    train_model.save_weights(DOMAIN_IDENTIFIER_PATH)


doms = pd.unique(read_item_data()['domain_id'])
def load_model():
    train_model = get_model()
    train_model.load_weights(DOMAIN_IDENTIFIER_PATH)
    return train_model
def predict_model(train_model,query_list,return_numeric=False,return_emb=False):
    """
        Returns prediction of train_model on batch of input
    """
    from tensorflow import TensorShape as ts
    import tensorflow.keras as keras
    
    
    x= np.reshape(np.array([preprocess_title(tit) for tit in query_list]),(-1,))
    
    x = get_emb_Kstr(x,x)
    
   
    

    input_batch = np.reshape(x,((-1,NUM_WORDS,512)))
    if return_emb:
        return input_batch
    
    
    #input_batch = np.concatenate(np.array([np.array([sentence_model.encode(preprocess_title(tit))]) for tit in query_list]),axis=0 )
    pred = train_model.predict(input_batch)
    #for i in range(pred.shape[0]):
    #    print((query_list[i],   doms[np.argmax(pred[i,:])])   ,np.max(pred[i,:]))
    if return_numeric:
        return pred
    else:
        return np.array([doms[np.argmax(pred[i,:])] for i in range(pred.shape[0]) if isinstance(doms[np.argmax(pred[i,:])],str ) ])

    
    return pred
    for i in range(pred.shape[0]):
        print((query_list[i],   doms[np.argmax(pred[i,:])])   ,np.max(pred[i,:]))
    
    
if __name__ == "__main__":
    #create_graph_domain()
    print(predict_model(load_model(),['RELOGIO SMARTWATCH','TV 40 polegadas','Jogo FIFA Xbox','Calça jeans','Placa NVIDIA']))

            
            
