'''
Created on 16 de nov de 2020

@author: klaus
'''
import jsonlines
from folders import DATA_DIR, SUBMISSIONS_DIR
import os
from os import path
import pandas as pd
import numpy as np

import igraph as ig
from input.read_input import read_item_data, get_mappings, NUM_DOMS
from nn import domain_string_identifier
from nn.domain_string_identifier import predict_model
from input.create_ratio import get_ratio

def create_graph_domain():
    """
        Creates graph linking (domain searched, domain bought)
    """
    
    """
        Fetch data
    """
    df = read_item_data()
    df['item_id'] = df.index
    dct_title = df['title'].to_dict()
    dct_domain = df['domain_id'].to_dict()
    dct_cat= df['category_id'].to_dict()
    
    dct_price = df['price'].to_dict()
    
    """ Ratio stuff """    
    from input.create_ratio import get_ratio
    dct_ratio_dom = get_ratio(which='domain_id')
    
    ratio_df = get_ratio(which='item_id',full=True)
    ratio_df['popularity'] = 100.0*ratio_df['bought'] + ratio_df['searched']
    dct_ratio_item_b = ratio_df['popularity'].to_dict()
    
    
    
    """
        JSON
    
    """
    check = lambda x: x <= np.round(413163*0.8).astype(np.int32)
    
    DATA_PATH = path.join(DATA_DIR,'train_dataset.jl')
    line_i = 0
    
    

    """
        Create graph vertices
    """
    g = ig.Graph() 
    
    counter, f_map_func, r_map_func = get_mappings()
    
    num_items = df.shape[0]
    for k in dct_title.keys():
        g.add_vertex(value=k,deg=dct_ratio_item_b[k],domain_id=dct_domain[k],price=dct_price[k],cat='item_id')

    """ ['item_id','domain_id','category_id','product_id'] """
    
    for k in pd.unique(df['domain_id']):
        g.add_vertex(value=k,cat='domain_id')


    for k in pd.unique(df['category_id']):
        g.add_vertex(value=k,cat='category_id')


    for k in pd.unique(df['product_id']):
        g.add_vertex(value=k,cat='product_id')

    
    
    """
        Create edges
    """
    E1 = []
    E2 = []
    
    with jsonlines.open(DATA_PATH) as reader:
        for line_i, obj in enumerate(reader):
            if check(line_i):
                print(line_i)
                L = []
                for h in obj['user_history']:
                    if h['event_type'] == 'view':
                        #print("Viewed {}".format(dct[h['event_info']]))
                        L.append(h['event_info'])
                    elif h['event_type'] == 'search':
                        #print("Searched {}".format(h['event_info']))
                        pass
                L_domain = [dct_domain[k] for k in L]
                L_domain = pd.unique(L_domain)
                L_cat = [dct_cat[k] for k in L]
                L_cat = pd.unique(L_cat)
                
                for i in range(len(L)):
                    E1.append(L[i])
                    E2.append(obj['item_bought'] )

    
    
    E1 = f_map_func['item_id'](E1)
    E2 = f_map_func['item_id'](E2)
    
    
    E =  pd.Series(list(zip(E1,E2))).value_counts()
    g.add_edges(E.index)
    g.es["weight"] = E.values
    
                     
    g.write_pickle(fname=path.join(DATA_DIR,'graph_item_to_item.pkl'))

def deg_matrix(W,pwr=1,flat=False, NA_replace_val = 1.0):
    import scipy.sparse
    """ Returns a diagonal matrix with the row-wise sums of a matrix W."""
    ws = W.sum(axis=0) if scipy.sparse.issparse(W) else np.sum(W,axis=0)
    D_flat = np.reshape(np.asarray(ws),(-1,))
    D_flat = np.power(D_flat,np.abs(pwr))
    is_zero = (D_flat == 0)
    if pwr < 0:
        D_flat[np.logical_not(is_zero)] = np.reciprocal(D_flat[np.logical_not(is_zero)])
        D_flat[is_zero] = NA_replace_val
    
    if scipy.sparse.issparse(W):
        

        if flat:
            return D_flat
        else:
            row  = np.asarray([i for i in range(W.shape[0])])
            col  = np.asarray([i for i in range(W.shape[0])])
            coo = scipy.sparse.coo_matrix((D_flat, (row, col)), shape=(W.shape[0], W.shape[0]))
            return coo.tocsr()
    else:
        if flat:
            return D_flat
        else:
            return(np.diag(D_flat))

def fit_RNN():
    import tensorflow as tf
    from tensorflow import keras
    import tf_geometric as tfg
    """
        Create graph
    """
    df = read_item_data()
    

    
    NUM_ITEMS = read_item_data().shape[0]
    NUM_FEATURES = 1

   


    
    

    counter, f_map_func, r_map_func = get_mappings()
        
    NUM_DOMS = pd.unique(df['domain_id']).shape[0]
    
    """ Load graph """
    G = ig.Graph.Read_Pickle(path.join(DATA_DIR,'graph_item_to_item.pkl'))
    #weights = np.log(1+np.array(G.es["weight"]))
    weights = np.array(G.es["weight"])
    
    indices = np.array([ np.array(e.tuple)  for e in G.es])
    indices = np.transpose(indices) 
    
    """ Create sparse matrix W """
    from scipy.sparse import coo_matrix
    import scipy.sparse
    row = indices[0,:]
    col = indices[1,:]
    
    W = coo_matrix((weights, (row, col)),shape=(NUM_ITEMS,NUM_ITEMS))
    """ Normalize rows """
    #W = deg_matrix(W,pwr=-1) @ W
    W = W.transpose()
    W = scipy.sparse.csr_matrix(W)
    assert scipy.sparse.issparse(W)
    
            
    
    @tf.function
    def smooth_labels(labels, factor=0.001):
        # smooth the labels
        labels = tf.cast(labels,tf.float32)
        labels *= (1 - factor)
        labels += (factor / tf.cast(tf.shape(labels)[1],tf.float32))
        # returned the smoothed labels
        return labels
    @tf.function
    def compute_loss(labels,logits):
        logits = tf.reshape(logits,(-1,NUM_ITEMS))
        labels = tf.reshape(labels,(-1,NUM_ITEMS))

        #logits = tf.nn.softmax(logits)
        #print(logits)
        
        logits = smooth_labels(logits)
        labels = smooth_labels(labels)
        
        losses = -tf.reduce_sum(logits*tf.math.log(labels),axis=1) 
        
        return tf.reduce_mean(losses)
    
    @tf.function
    def evaluate(labels,logits):
        logits = tf.reshape(logits,(-1,NUM_ITEMS))
        labels = tf.reshape(labels,(-1,NUM_ITEMS))

        #logits = tf.nn.softmax(logits)
        #print(logits)
        
        logits = smooth_labels(logits)
        labels = smooth_labels(labels)
        
        acc = tf.metrics.categorical_accuracy(labels,logits)
        
        return tf.reduce_mean(acc)
    
    
    
    """
        Read data, yadda yadda
    
    """
    from input.create_ratio import get_ratio
    ratio_df = get_ratio(which='item_id',full=True)
    ratio_df['popularity'] = 100.0*ratio_df['bought'] + ratio_df['searched']
    dct_ratio_item_b = ratio_df['popularity'].to_dict()
    

    dct = df['title'].to_dict()
    dct_domain = df['domain_id'].to_dict()
    dct_cat = df['category_id'].to_dict()
    dct_price = df['price'].to_dict()
    
  
    
    """ Ratio stuff """    
    from input.create_ratio import get_ratio
    category_df = get_ratio(which='category_id',full=True)
    domain_df = get_ratio(which='domain_id', full = True)
    
  
    
    feat_1, feat_2, feat_3 = domain_df['searched'].to_dict(), domain_df['bought'].to_dict(), domain_df['rat'].to_dict()
    
    feat_1,feat_2,feat_3 = [ [X[dct_domain[k]] for k in df.index]  for X in [feat_1,feat_2,feat_3]]
    
    feat_1_1, feat_2_1, feat_3_1 = category_df['searched'].to_dict(), category_df['bought'].to_dict(), category_df['rat'].to_dict()
    feat_1_1,feat_2_1,feat_3_1 = [ [X[dct_cat[k]] for k in df.index]  for X in [feat_1_1,feat_2_1,feat_3_1]]
    
    
    
    def standardize(x):
        return (x - np.min(x)) / (np.max(x)+1e-06 - np.min(x))
    
    feat_1, feat_2, feat_3 = [standardize(x) for x in [feat_1,feat_2,feat_3]]
    
    feat_1_1, feat_2_1, feat_3_1 = [standardize(x) for x in [feat_1_1,feat_2_1,feat_3_1]]
    
    del df
    del domain_df
    del category_df
    del G    
    #dom_ratios = np.array([dct_ratio_dom[k] for k in pd.unique(df['domain_id'].values)])
    #dom_ratios = (dom_ratios - np.mean(dom_ratios)) / np.std(dom_ratios)

    
    
    from  nn.domain_string_identifier import load_model
    domain_prediction_model = load_model()
    def my_generator(mode='train'):
            if mode == 'train':
                check = lambda x: x <= np.round(413163*0.8).astype(np.int32)
            elif mode == 'val':
                check = lambda x: x > np.round(413163*0.8).astype(np.int32)
            else:
                check = lambda x: True
            DATA_PATH = path.join(DATA_DIR,'test_dataset.jl' if mode == 'test' else 'train_dataset.jl')
            print("Reading....")
            
            X = np.zeros((NUM_ITEMS,10)).astype(np.float32)
            with jsonlines.open(DATA_PATH) as reader:
                for line_i, obj in enumerate(reader):
                    
                    if check(line_i):
                        L = []
                        S = []
                        C =[]
                        IDS = []
                        for h in obj['user_history']:
                            if h['event_type'] == 'view':
                                L.append(dct_domain[h['event_info']])
                                C.append(dct_cat[h['event_info']])
                                IDS.append(h['event_info'])
                                
                            elif h['event_type'] == 'search':
                                S.append(h['event_info'])
    
                        
                        if  obj['item_bought'] in L:
                            continue
                        
                        
                        L =  f_map_func['domain_id'](L)
                        C =  f_map_func['category_id'](C)
                        IDS_map =  f_map_func['item_id'](IDS)
                        
                        """ Adjust graph """
                        Y = np.zeros((NUM_ITEMS,1)).astype(np.float32)
                        
                        """
                        X[:,0] = feat_1
                        X[:,1] = feat_2
                        X[:,2] = feat_3
                        X[:,6] = feat_1_1
                        X[:,7] = feat_2_1
                        X[:,8] = feat_3_1
                        
                        #if len(S) > 0:
                        #    X[:,8] =  np.mean(predict_model(domain_prediction_model,S,return_numeric=True),axis=0)
                        """
                        target_id =  f_map_func['item_id']( [ obj['item_bought'] ] )[0]
                        if not mode == 'test':
                            Y[    target_id,0    ] = 1.0
                        """
                        for i,k in enumerate(IDS_map):
                            X[k,3] +=  1
                            X[k,4] +=  dct_ratio_item_b[IDS[i]]/len(C)
                            X[k,5] =  dct_price[IDS[i]]
                        
                        #W[target_id,:] = (np.clip(np.array(W[target_id,:].todense())-1,a_min=0.0,a_max=None))
                        X[:,9] = np.reshape(np.asarray(W @ X[:,3]),(-1,))
                        X[:,9] = X[:,8] * X[:,2]
                        #X[:,:8] = 0

                        for i in range(10):
                            X[:,i] = (X[:,i] - np.min(X[:,i])) / (1e-06+ np.max(X[:,i]) - np.min(X[:,i])) 
                        """
                        if not mode == 'test':
                            Y[    target_id,0    ] = 0.0
                        #X = X -0.5
                        yield X,Y
                    
    """
        Optimize
    """

    BS = 2
    step = 0
    
    
    def batch_generator(mode, loop =True,batch_size=BS):
        BATCH_X = []
        BATCH_Y = []
        i = 0
        while True:
            for x,y in my_generator(mode):
                
                BATCH_X.append(x[None,:,:])
                BATCH_Y.append(y[None,:,:])
                i+= 1
                if i % batch_size == 0:      
                    yield np.concatenate(BATCH_X,axis=0), np.concatenate(BATCH_Y,axis=0)
                    BATCH_X = []
                    BATCH_Y = []
                    i = 0 
            if loop == False:
                yield np.concatenate(BATCH_X,axis=0), np.concatenate(BATCH_Y,axis=0)
                break
    """
        Define train_model
    """
    import  tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    inp_x = keras.Input((NUM_ITEMS,10))
    x = layers.Dense(32,activation='relu')(inp_x)
    x = layers.Dense(32,activation='relu')(x)
    x = layers.Dense(1)(x)
    x = layers.Flatten()(x)
    x = layers.Softmax(axis=-1)(x)
    
    train_model = keras.Model(inputs=[inp_x],outputs=[x])
    print(train_model.summary())
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.5*1e-2,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.2*1e-2)
    
    train_model.compile(optimizer=optimizer,loss=compute_loss,metrics=[evaluate])
    from functools import partial
    from input.read_input import TRAIN_LINES
    train_model.fit_generator(batch_generator('train',True),
              steps_per_epoch=TRAIN_LINES//BS,
              epochs=1
              )
    
    ITEM_PATH = path.join(DATA_DIR,'train_model','item_classifier.h5')
    train_model.save_weights(ITEM_PATH)
    
    def predict(mode):
        PREDS = []
        CONFS = []
        NUM_SELECT = 10
        batch_size = 1
        for batch_id, X in enumerate(batch_generator(mode,batch_size=batch_size,loop=False)):
            x = X[0]
            print("Predicting {} - Batch {}".format(mode,batch_id))
            pred = train_model.predict_on_batch(x)
            if batch_id == 0:
                print(pred)
            PREDS.append(tf.argsort(pred,axis=-1)[:,-NUM_SELECT:])
            CONFS.append(tf.sort(pred,axis=-1)[:,-NUM_SELECT:])
            
        PREDS = np.concatenate(PREDS,axis=0)
        CONFS = np.concatenate(CONFS,axis=0)
        #PREDS = np.concatenate([PREDS,CONFS],axis=1)
        cols = ['pred_{}'.format(k) for k in range(NUM_SELECT)] 
        fname = os.path.join(DATA_DIR,'item_pred_{}.csv'.format(mode))
        pd.DataFrame(PREDS,index=range(PREDS.shape[0]),columns=cols).to_csv(fname)
    
    predict('train')
    predict('val')
    predict('test')
    
    

#############################################################################################################################################################



if __name__ == "__main__":
    #create_graph_domain()
    fit_RNN()

            
            
