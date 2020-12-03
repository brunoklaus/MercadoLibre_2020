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
from nn import domain_string_identifier
from nn.domain_string_identifier import predict_model
from input.create_ratio import get_ratio, load_language_df


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

def train_neural_domain_prediction():
    import tensorflow as tf
    """
        Create graph
    """
    from input.read_input import read_item_data
    df = read_item_data()
    dct_condition = df['condition'].to_dict()
    
    df2 = load_language_df()
    dct_lan_pt = df2['score_pt'].to_dict()
    dct_lan_en = df2['score_en'].to_dict()
    dct_lan_es = df2['score_es'].to_dict()
    
    
    
    NUM_ITEMS = read_item_data().shape[0]
    NUM_FEATURES = 1

   


    
    
    
    from input.read_input import  get_mappings, NUM_DOMS
    counter, f_map_func, r_map_func = get_mappings()
        
    NUM_DOMS = pd.unique(df['domain_id']).shape[0]
    NUM_CATS = pd.unique(df['category_id']).shape[0]
    
    """ Load graph """
    graph_fname = path.join(DATA_DIR,'graph_domain_to_domain.pkl')
    if not path.isfile(graph_fname):
        input("Did not find graph at {}. Will have to create it from scratch... (Any key to continue)".format(graph_fname))
        G = create_graph_domain()
    else:
        G = ig.Graph.Read_Pickle(path.join(DATA_DIR,'graph_domain_to_domain.pkl'))
    #weights = np.log(1+np.array(G.es["weight"]))
    weights = np.array(G.es["weight"])
    
    indices = np.array([ np.array(e.tuple)  for e in G.es]) - NUM_ITEMS 
    indices = np.transpose(indices) 
    
    """ Create sparse matrix W """
    from scipy.sparse import coo_matrix
    import scipy.sparse
    row = indices[0,:]
    col = indices[1,:]
    
    W = coo_matrix((weights, (row, col)),shape=(NUM_DOMS,NUM_DOMS))
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
        logits = tf.reshape(logits,(-1,NUM_DOMS))
        labels = tf.reshape(labels,(-1,NUM_DOMS))

        #logits = tf.nn.softmax(logits)
        #print(logits)
        
        logits = smooth_labels(logits)
        labels = smooth_labels(labels)
        
        losses = -tf.reduce_sum(logits*tf.math.log(labels),axis=1) 
        
        return tf.reduce_mean(losses)
    
    @tf.function
    def evaluate(labels,logits):
        logits = tf.reshape(logits,(-1,NUM_DOMS))
        labels = tf.reshape(labels,(-1,NUM_DOMS))

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
    ratio_df = get_ratio(which='item_id',full=True,alternate=False)
    dct_ratio_item_b = ratio_df['bought'].to_dict()
    dct_ratio_item_s = ratio_df['searched'].to_dict()
    dct_ratio_item_r = ratio_df['searched'].to_dict()
    

    dct = df['title'].to_dict()
    dct_domain = df['domain_id'].to_dict()
    dct_cat = df['category_id'].to_dict()
    dct_price = df['price'].to_dict()
    
  
    
    """ Ratio stuff """    
    from input.create_ratio import get_ratio
    category_df = get_ratio(which='category_id',full=True)
    domain_df = get_ratio(which='domain_id', full = True)
    
  
    
    feat_1, feat_2, feat_3 = domain_df['searched'].values, domain_df['bought'].values, domain_df['rat'].values
    
    feat_4, feat_5 = domain_df['out_bought'].values,domain_df['rat2'].values
    
    feat_1_1, feat_2_1, feat_3_1 = category_df['searched'].values, category_df['bought'].values, category_df['rat'].values
    
    
    def standardize(x):
        return (x - np.min(x)) / (np.max(x)+1e-06 - np.min(x))
    
    feat_1, feat_2, feat_3 = [standardize(x) for x in [feat_1,feat_2,feat_3]]
    
    feat_1_1, feat_2_1, feat_3_1 = [standardize(x) for x in [feat_1_1,feat_2_1,feat_3_1]]
    
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
    
                        
                        
                        
                        L =  f_map_func['domain_id'](L)
                        C =  f_map_func['category_id'](C)
                        
                        df = pd.DataFrame(
                            {"domain_id":L,
                             "feat_1_1":[feat_1_1[C[i]-NUM_ITEMS-NUM_DOMS] for i in range(len(L))],
                             "feat_2_1":[feat_2_1[C[i]-NUM_ITEMS-NUM_DOMS] for i in range(len(L))],
                             "feat_3_1":[feat_3_1[C[i]-NUM_ITEMS-NUM_DOMS] for i in range(len(L))],
                             
                             },
                            index=IDS)
                        
                        
                        df['recency'] = range(len(L))
                        df['freq'] = np.ones((len(L),))
                        df['price'] = [ dct_price[k] for k in IDS]
                        df['item_b'] =[ dct_ratio_item_b[k] for k in IDS]
                        df['item_s'] =[ dct_ratio_item_s[k] for k in IDS]
                        
                        df['condition'] =[dct_condition[k] for k in IDS]
                        df['lan_pt'] = [dct_lan_pt[k] for k in IDS]
                        df['lan_en'] = [dct_lan_en[k] for k in IDS]
                        df['lan_es'] = [dct_lan_es[k] for k in IDS]
                        
                        
                        """ Adjust graph """
                        Y = np.zeros((NUM_DOMS,1)).astype(np.float32)
                        X = np.zeros((NUM_DOMS,55+55)).astype(np.float32)
                        
                        
                        X[:,0] = feat_1
                        X[:,1] = feat_2
                        X[:,2] = feat_3
                        X[:,3] = feat_4

                        i=4
                        for g, df2 in df.groupby(["domain_id"]):
                            i=4
                            v = df2.to_numpy()[:,1:]                            
                            X[g-NUM_ITEMS,i:i+(v.shape[1])] = np.sum(v,axis=0)
                            i += v.shape[1]
                            X[g-NUM_ITEMS,i:i+(v.shape[1])] = np.mean(v,axis=0)
                            i += v.shape[1]
                            X[g-NUM_ITEMS,i:i+(v.shape[1])] = np.nanstd(v,axis=0)
                            i += v.shape[1]
                            X[g-NUM_ITEMS,i:i+(v.shape[1])] = np.max(v,axis=0)
                            i += v.shape[1]

                        

                        if len(S) > 0:
                            s_pred = predict_model(domain_prediction_model,S,return_numeric=True)
                        else:
                            s_pred = np.zeros_like((1,NUM_DOMS))
                        if len(S) > 0:
                            X[:,i] =  np.mean(s_pred,axis=0)
                            X[:,i+1] =  np.max(s_pred,axis=0)
                            try:
                                X[:,i+2] =  np.nanstd(s_pred,axis=0)
                            except:
                                X[:,i+2] =  X[:,i+2] 
                            i += 3
                        
                        X[:,55:] = np.reshape(np.asarray(W @ X[:,55:]),(-1,X.shape[1]-55))
                        if not mode == 'test':
                            Y[     f_map_func['domain_id']( [ dct_domain[obj['item_bought']] ] )[0] - NUM_ITEMS,0    ] = 1.0
                        
                        
                        #X[:,:8] = 0

                        for i in range(55+3):
                            X[:,i] = (X[:,i] - np.min(X[:,i])) / (1e-06+ np.max(X[:,i]) - np.min(X[:,i])) 
                        
                        #X = X -0.5
                        yield X,Y
                    
    """
        Optimize
    """

    BS = 64
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
        Define model
    """
    import  tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    inp_x = keras.Input((NUM_DOMS,55+55))
    x = layers.Dense(64,activation='relu')(inp_x)
    x = layers.Dense(64,activation='relu')(x)
    x = layers.Dense(64,activation='relu')(x)
    x = layers.Dense(1)(x)
    x = layers.Flatten()(x)
    x = layers.Softmax(axis=-1)(x)
    
    model = keras.Model(inputs=[inp_x],outputs=[x])
    print(model.summary())
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.5*1e-2,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1*1e-2)
    
    model_fname = path.join(DATA_DIR,'model',"NEURAL_DOMAIN_PRED.h5")
    model.compile(optimizer=optimizer,loss=compute_loss,metrics=[evaluate])
    from functools import partial
    from input.read_input import TRAIN_LINES
    
    #model.load_weights(path.join(DATA_DIR,"MY_MODEL_2.h5"))
    if not path.isfile(model_fname):
        input("Warning!!! Did not find model weights at {}. Training takes many, many, many hours! (Press ENTER)".format(model_fname))
        
        model.fit_generator(batch_generator('train',True),
                  steps_per_epoch=TRAIN_LINES//BS,
                  epochs=5
                  )
        model.save_weights(model_fname)

    else:
        model.load_weights(model_fname)
        print("Testing fit... should be about 0.41 to 0.45")
        model.fit_generator(batch_generator('train',True),
          steps_per_epoch=25,
          epochs=1
          )

    
    
    def predict(mode):
        PREDS = []
        CONFS = []
        NUM_SELECT = 10
        batch_size = 320
        for batch_id, X in enumerate(batch_generator(mode,batch_size=batch_size,loop=False)):
            x = X[0]
            print("Predicting {} - Batch {}".format(mode,batch_id))
            pred = model.predict_on_batch(x)
            if batch_id == 0:
                print(pred)
            PREDS.append(tf.argsort(pred,axis=-1)[:,-NUM_SELECT:])
            CONFS.append(tf.sort(pred,axis=-1)[:,-NUM_SELECT:])
            
        PREDS = np.concatenate(PREDS,axis=0)
        CONFS = np.concatenate(CONFS,axis=0)
        PREDS = np.concatenate([PREDS,CONFS],axis=1)
        cols = ['pred_{}'.format(k) for k in range(NUM_SELECT)] + \
         ['conf_{}'.format(k) for k in range(NUM_SELECT)] 
        fname = os.path.join(DATA_DIR,'dom_pred_{}.csv'.format(mode))
        pd.DataFrame(PREDS,index=range(PREDS.shape[0]),columns=cols).to_csv(fname)
    
    predict('val')
    predict('test')
    predict('train')
    
    

#############################################################################################################################################################


def create_graph_domain():
    """
        Creates graph linking (domain searched, domain bought)
    """
    
    """
        Fetch data
    """
    
    from input.read_input import read_item_data
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
    from input.read_input import get_mappings
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
                        E1.append(dct_domain[L[i]])
                        E2.append(dct_domain[obj['item_bought']] )

    
    
    E1 = f_map_func['domain_id'](E1)
    E2 = f_map_func['domain_id'](E2)
    
    
    E =  pd.Series(list(zip(E1,E2))).value_counts()
    g.add_edges(E.index)
    g.es["weight"] = E.values
    
                     
    g.write_pickle(fname=path.join(DATA_DIR,'graph_domain_to_domain.pkl'))



if __name__ == "__main__":
    train_neural_domain_prediction()

            
            
