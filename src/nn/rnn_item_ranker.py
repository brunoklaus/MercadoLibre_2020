'''
Created on 11 de nov de 2020

@author: klaus
'''
import jsonlines
from folders import DATA_DIR, SUBMISSIONS_DIR
import os
from os import path
import pandas as pd
import numpy as np
import igraph as ig
from input.read_input import read_item_data, ndcg, first_two_words
import tensorflow as tf
from nn.domain_string_identifier import predict_model
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


from functools import partial

SEQ_LEN=30
BATCH_SIZE= 64

NUM_UNITS = 32

CANDIDATES = 30


def read_predictions(mode='train'):
    assert mode in ['train','val','test']
    if mode in ['train','val']:
        df1 = pd.read_csv(path.join(DATA_DIR,'RNN_{}.csv'.format('train')),index_col=0)
        
        df1 = df1.iloc[0:330531,:]
        df2 = pd.read_csv(path.join(DATA_DIR,'RNN_{}.csv'.format('val')),index_col=0)
        df2 = df2.iloc[0:82632,:]
        return pd.concat([df1,df2],axis=0,ignore_index=True)
    elif mode == 'test':
        return pd.read_csv(path.join(DATA_DIR,'RNN_{}.csv'.format('test')),index_col=0)
    
USE_EMB = True
EMB_SIZE = 512*2 if USE_EMB else 0
ATTR_SIZE = 3 + 7 + 11

def fix_na(x):
    if np.isnan(x):
        return 0
    else:
        return x

def meli_iterator(mode = 'train',batch_size = BATCH_SIZE,full=False):
    from input.read_input import get_sentence_model, get_emb
     
    from input.create_ratio import load_language_df
    TRAIN_LINES = 413163
    TEST_LINES = 177070
    df = read_item_data()
    
    dct_condition = df['condition'].to_dict()

    df2 = load_language_df()
    dct_lan_pt = df2['score_pt'].to_dict()
    dct_lan_en = df2['score_en'].to_dict()
    dct_lan_es = df2['score_es'].to_dict()
    
    dct = df['title'].to_dict()
    dct_domain = df['domain_id'].to_dict()
    dct_cat = df['category_id'].to_dict()
    dct_price = df['price'].to_dict()

    """ Ratio stuff """    
    from input.create_ratio import get_ratio
    dct_ratio_dom = get_ratio(which='domain_id')
    
    ratio_df = get_ratio(which='item_id',full=True)
    ratio_df['popularity'] = 100.0*ratio_df['bought'] + ratio_df['searched']
    dct_ratio_item_p = ratio_df['popularity'].to_dict()
    
    ratio_df = get_ratio(which='item_id',full=True,alternate=False)
    dct_ratio_item_b = ratio_df['bought'].to_dict()
    dct_ratio_item_s = ratio_df['searched'].to_dict()
    dct_ratio_item_r = ratio_df['rat'].to_dict()
    
    
    
    df['item_bought'] = [dct_ratio_item_b[k] for k in df.index]

    
    
    dct_ratio_cat = get_ratio(which='category_id',full=True)
    dct_ratio_cat_s, dct_ratio_cat_b, dct_ratio_cat  = dct_ratio_cat['searched'].to_dict(),\
                                                       dct_ratio_cat['bought'].to_dict(),\
                                                       dct_ratio_cat['rat'].to_dict(),\
                                                       
                                                       
    
    dct_ratio_dom = get_ratio(which='domain_id',full=True)
    dct_ratio_dom_s, dct_ratio_dom_b, dct_ratio_dom  = dct_ratio_dom['searched'].to_dict(),\
                                               dct_ratio_dom['bought'].to_dict(),\
                                               dct_ratio_dom['rat'].to_dict(),\
    
    
    dct_ratio_item = get_ratio(which='item_id')
    
    
    dct_domain_df = {}
    dct_cat_df = {}
    for dom, df2 in df.groupby('domain_id'):
        df2= df2.sort_values(['item_bought'],ascending=False)#.iloc[0:10,:]
        dct_domain_df[dom] =  df2
    
    for cat, df2 in df.groupby('category_id'):
        df2 = df2.sort_values(['item_bought'],ascending=False)#.iloc[0:10,:]
        dct_cat_df[cat] =  df2
    del df
    del df2
            
    def _begin_overfit_avoid(L_k):
    
        if not mode == 'train':
            return
        target_item = obj['item_bought']
        target_dom = dct_domain[obj['item_bought']]
        target_cat = dct_cat[obj['item_bought']]
        for this_item in L_k:
            """ Bought """
            if this_item == target_item:
                assert dct_ratio_item_b[this_item] > 0
                dct_ratio_item_b[this_item] -= 1
            """ Search """
            dct_ratio_item_s[this_item] -= 1
            assert dct_ratio_item_s[this_item] >= 0
            
            """ Ratio """
            dct_ratio_item_r[this_item] = dct_ratio_item_b[this_item] / (dct_ratio_item_s[this_item]+1)
        for this_dom in pd.unique([dct_domain[k] for k in L_k]):
            if not isinstance(this_dom,str):
                continue
            
            """ Bought """
            if this_dom == target_dom:
                assert dct_ratio_dom_b[this_dom] > 0
                dct_ratio_dom_b[this_dom]  -= 1
            """ Search """
            dct_ratio_dom_s[this_dom] -= 1
            assert dct_ratio_dom_s[this_dom] >= 0
            """ Ratio """
            dct_ratio_dom[this_dom] = dct_ratio_dom_b[this_dom] / (dct_ratio_dom_s[this_dom]+1)
        for this_cat in pd.unique([dct_cat[k] for k in L_k]):
                
            """ Bought """
            if this_cat == target_cat:
                assert dct_ratio_cat_b[this_cat] > 0
                dct_ratio_cat_b[this_cat] -= 1
            """ Search """
            dct_ratio_cat_s[this_cat] -= 1
            assert dct_ratio_cat_s[this_cat] >= 0
            """ Ratio """
            dct_ratio_cat[this_cat] = dct_ratio_cat_b[this_cat] / (dct_ratio_cat_s[this_cat]+1)
        
    def _end_overfit_avoid(L_k):
            if not mode == 'train':
                return
            target_item = obj['item_bought']
            target_dom = dct_domain[obj['item_bought']]
            target_cat = dct_cat[obj['item_bought']]
            for this_item in L_k:
                """ Bought """
                if this_item == target_item:
                    #assert dct_ratio_item_b[this_item] >= 0
                    dct_ratio_item_b[this_item] += 1
                """ Search """
                #assert dct_ratio_item_s[this_item] >= 0
                dct_ratio_item_s[this_item] += 1
                
                
                """ Ratio """
                dct_ratio_item_r[this_item] = dct_ratio_item_b[this_item] / (dct_ratio_item_s[this_item]+1)
            for this_dom in pd.unique([dct_domain[k] for k in L_k]):
                if not isinstance(this_dom,str):
                    continue

                """ Bought """
                if this_dom == target_dom:
                    #assert dct_ratio_dom_b[this_dom] >= 0
                    dct_ratio_dom_b[this_dom]  += 1
                """ Search """
                #assert dct_ratio_dom_s[this_dom] >= 0
                dct_ratio_dom_s[this_dom] += 1
                """ Ratio """
                dct_ratio_dom[this_dom] = dct_ratio_dom_b[this_dom] / (dct_ratio_dom_s[this_dom]+1)
            for this_cat in pd.unique([dct_cat[k] for k in L_k]):
                """ Bought """
                if this_cat == target_cat:
                    #assert dct_ratio_cat_b[this_cat] >= 0
                    dct_ratio_cat_b[this_cat] += 1
                """ Search """
                #assert dct_ratio_cat_s[this_cat] >= 0
                dct_ratio_cat_s[this_cat] += 1
                """ Ratio """
                dct_ratio_cat[this_cat] = dct_ratio_cat_b[this_cat] / (dct_ratio_cat_s[this_cat]+1)
    
    
    if mode == 'train':
        check = lambda x: x <= np.round(413163*0.8).astype(np.int32)
    elif mode == 'val':
        check = lambda x: x > np.round(413163*0.8).astype(np.int32)
    else:
        check = lambda x: True
    
    DATA_PATH = path.join(DATA_DIR,'test_dataset.jl' if mode == 'test' else 'train_dataset.jl')
    
    def rank_to_order(L,rank):
        assert rank.shape[0] == L.shape[0]
        return L[(-rank).argsort(kind='mergesort')]
    
    pred = {}
    
    actual = []
    
    X = []
    Y = []
    MASK = []
    LKS = []
    ACTUAL = []
    while True:
        with jsonlines.open(DATA_PATH) as reader:
            print("Start!!!")
            for line_id, obj in enumerate(reader):
                if check(line_id):
                    #print(i)
                    L = []
                    timestamps =[]
                    dct_emb = {}
                    if mode == 'test':
                        obj['item_bought'] = -999
                    for h in obj['user_history']:
                        if h['event_type'] == 'view':
                            L.append(h['event_info'])
                            timestamps.append(pd.Timestamp(h['event_timestamp']))
                        elif h['event_type'] == 'search':
                            pass
                    
                    def divide_time(d):
                        d = pd.Timedelta(d).total_seconds()
                        MINUTE_M = 60
                        HOUR_M = MINUTE_M*60
                        DAY_M = HOUR_M*24
                        
                        div = [1,24,60]
                        res = [0,0,0]
                        for i, M in enumerate([DAY_M,HOUR_M,MINUTE_M]):
                            res[i] = np.floor(d/M)
                            d -= M*res[i]
                            res[i] /= div[i]
                            #res[i] -= 0.5
    
                        return tuple(res)
                    
                    if not full and len(L) < 2:
                        continue
                    
                    """ Create attributes """
                    if len(L) == 0:
                        attrs = np.zeros((1,(CANDIDATES+1)+ATTR_SIZE+EMB_SIZE))
                        targets = np.zeros((1,(CANDIDATES+1)))
                        targets[0,-1] = 0
                        L_k = []
                    else:
                        delta = [   timestamps[-1]-timestamps[i] for i in range(0,len(timestamps))]
                        
                        """
                            We'll use the latest delta
                        """
                        L = L[::-1]
                        u, unique_id = np.unique(np.array(L), return_index=True)
                        
                        #delta_day, delta_hour, delta_minute = zip(*[divide_time(d) for d in delta])
                        deltas = np.array([divide_time(d) for d in delta])
                        deltas = deltas[unique_id][:SEQ_LEN]
                        
                        L_k = np.array(L)[unique_id][:CANDIDATES]
                        _begin_overfit_avoid(L_k)
                        
                        """
                            rank_freq initial calculation needs whole L
                        """
                        rank_freq = pd.Series(L,index=range(len(L))).value_counts(sort=False,normalize=True)
                        rank_freq = rank_freq.rank(method="average").to_dict()
                        
                        
                        L = np.array(L)[unique_id][:SEQ_LEN]
                        """
                            Calculate ranks
                        """
                        condition = np.array([1.0 if dct_condition[k] == 'new' else  0.0 for k in L])[:,None]
                        
                        #ratio_dom = np.array([dct_ratio_dom[dct_domain[k]] for k in L])[:,None]
                        #ratio_cat = np.array([dct_ratio_cat[dct_cat[k]] for k in L])[:,None]
                        #ratio_item = np.array([dct_ratio_item[k] for k in L])[:,None]
                        price = np.log(np.array([1 + np.abs(fix_na(dct_price[k])) for k in L])[:,None])
                        rank_freq = np.array([rank_freq[k] for k in L])[:,None]
                        #rank_latest = (1.0 - np.arange(len(L))/len(L))[:,None]
                        
                        rank_ratio_dom = pd.Series([dct_ratio_dom[dct_domain[k]] for k in L_k]).rank(method="average").to_numpy()
                        rank_ratio_cat = pd.Series([dct_ratio_cat[dct_cat[k]] for k in L_k]).rank(method="average").to_numpy()
                        rank_ratio_item = pd.Series([dct_ratio_item_r[k] for k in L_k]).rank(method="average").to_numpy()
                        rank_latest = (1.0 - np.arange(len(L))/len(L))
                        
                        x = []
                        x.append([dct_ratio_dom[dct_domain[k]] for k in L_k])
                        x.append(rank_ratio_dom)
                        x.append(rank_ratio_cat)
                        x.append(rank_ratio_item)
                        
                                 
                                    
                        
                        x.append([dct_ratio_dom[dct_domain[k]] for k in L_k])
                        
                        x.append([dct_ratio_cat[dct_cat[k]] for k in L_k])
                        x.append([dct_ratio_item_b[k] for k in L_k])
                        x.append([dct_ratio_item_s[k] for k in L_k])
                        x.append([dct_ratio_item_r[k] for k in L_k])
                        x.append(list(rank_latest/len(L_k)))
                        x.append([-dct_price[k] for k in L_k])
                        
                        x.append([-dct_condition[k] for k in L_k])
                        x.append([-dct_lan_en[k] for k in L_k])
                        x.append([-dct_lan_es[k] for k in L_k])
                        x.append([-dct_lan_pt[k] for k in L_k])
                        
                        
                        """
                            
                        """
                        #true_val = (np.array([str(dct_domain[k]) for k in L]) == dct_domain[obj['item_bought']])
                        #true_val = np.logical_and(true_val,[k in L_k for k in L])
                        #true_val = true_val[:,None]
                        #true_val = np.ones_like(true_val)
                        #true_val = np.random.rand(*(true_val.shape))
                        
                        
                        
                        
                        assert all([k in L for k in L_k])
                        
                        ids = [np.where(L_k == l)[0][0] if l in L_k else CANDIDATES for l in L ]
                        ids_onehot = np.zeros((len(L),(CANDIDATES+1)))
                        ids_onehot[np.arange(len(L)),ids] = 1
                        #ids_onehot = ids_onehot[:,0:10]
                        
                        

                        

                        """
                            Create numeric attributes plus embeddings
                        """

                        attr_list = [ids_onehot,deltas,condition,price,rank_freq] + [np.array(_x)[:,None] for _x in x]
                        if USE_EMB:                 
                            emb = predict_model(get_sentence_model(),query_list=[dct[k] for k in L_k],return_emb=True)
                            emb = np.reshape(emb[:,0:(EMB_SIZE//512),:],(emb.shape[0], EMB_SIZE))
                            attr_list.append(emb)

                        attrs  = np.concatenate(attr_list,axis=1)
                        
                        """ Create targets """
                        if mode == 'test':
                            targets = np.zeros((1,(CANDIDATES+1)))
                        else:
                            _b1 = (np.array(list(L_k == obj['item_bought'])))
                            _b2 = (np.array(list([str(dct_domain[k]) for k in L_k])) == dct_domain[obj['item_bought']])
        
                            targets =     _b1.astype(np.float32)*1.0  #+ _b2.astype(np.float32)*0.0 
                            if np.sum(targets) == 0:
                                targets = np.zeros((1,(CANDIDATES+1)))
                                targets[0,-1] = 1                            
                                if not full:
                                    _end_overfit_avoid(L_k)
                                    continue
                            else:
                                targets = np.array(targets)/np.sum(targets)
                                targets = np.concatenate([targets[None,:],np.zeros((1,CANDIDATES+1-len(L_k)))],axis=1)

                    """ Add attributes, targets. """
                    if attrs.shape[0] < SEQ_LEN:
                        attrs = np.concatenate([np.zeros((SEQ_LEN-attrs.shape[0],attrs.shape[1],) ),attrs],axis=0)
                    attrs =attrs[-SEQ_LEN:,:]
                    attrs = attrs.astype(np.float32)
                    _end_overfit_avoid(L_k)
                    

                    X.append(attrs[None,:])
                    Y.append(targets)
                    mask = np.concatenate([np.ones((len(L_k))),
                                            np.zeros((CANDIDATES+1)-len(L_k))]).astype(np.float32)[None,:]
                    MASK.append(mask)
                    
                    
                    LKS.append(np.concatenate([L_k,-1*np.ones( ((CANDIDATES+1)-len(L_k),) )]  )[None,:] )
                    ACTUAL.append(np.array([obj['item_bought'] ])[None,:])
                    
                    
                if len(X) == batch_size :
                    
                    X = np.concatenate(X,axis=0)
                    Y = np.concatenate(Y,axis=0)

                    MASK = np.concatenate(MASK,axis=0)
                    LKS = np.concatenate(np.array(LKS).astype(np.int32),axis=0)
                    ACTUAL = np.concatenate(np.array(ACTUAL).astype(np.int32),axis=0)

                    
                    yield (X,MASK,LKS,ACTUAL),Y
                    X = []
                    Y = []
                    MASK = []
                    LKS = []
                    ACTUAL = []
    
                    #print(attrs.shape)
        if full:
            check = (lambda i: True)
                #L.append(obj)
import tensorflow.keras.backend as K
@tf.function
def func(t):
    t= t[0]*t[1]
    t = tf.math.divide_no_nan(t,K.sum(t,axis=0))
    return t

def train_model():
    
    from tensorflow.keras.layers.experimental import preprocessing
    
    BIDIR = False
    #import tensorflow.keras.backend as K
    import tensorflow.keras as keras
    import tensorflow.keras.layers as l
    input_X = keras.Input(shape=(SEQ_LEN,ATTR_SIZE+EMB_SIZE+(CANDIDATES+1)),name='input_x')
    
    #input_onehot = l.Lambda(lambda t: t[:,:,0:(CANDIDATES+1)])(input_X)
    @tf.function
    def select(t):
        return t[:,:,(CANDIDATES+1):]
    print(input_X.shape)
    
    input_onehot = l.Lambda(lambda t: t[:,:,:(CANDIDATES+1)],name='input_onehot')(input_X)
    x = l.Lambda(lambda t: t[:,:,(CANDIDATES+1):],name='input_attr_and_emb')(input_X)
    x_attr = l.Lambda(lambda t: t[:,:,:ATTR_SIZE],name='input_attr')(x)
    x_emb = l.Lambda(lambda t: t[:,:,ATTR_SIZE:],name='input_emb')(x)
    
    """ Attr """
    x = tf.keras.layers.BatchNormalization()(x_attr)
    x = tf.keras.layers.Dense(128,activation='relu')(x)    
    x = l.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16,activation='relu')(x)
    x_attr = x
    
    
    
    
    #x = preprocessing.Normalization()(x)
    """ EMB """
    x = tf.keras.layers.BatchNormalization()(x_emb)
    x = tf.keras.layers.Dense(128,activation='relu')(x)    
    x = l.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64,activation='relu')(x)
    x = l.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16,activation='relu')(x)
    x_emb = x
    
    x = l.Concatenate()([x_emb,x_attr])
    
    
    

    #input_MASK = keras.Input(shape=((CANDIDATES+1),),name='input_mask')



    if BIDIR:
        x = l.Bidirectional(l.LSTM(units=NUM_UNITS,return_sequences=True),merge_mode='ave')(x)
        x = l.Bidirectional(l.LSTM(units=NUM_UNITS,return_sequences=True),merge_mode='ave')(x)
        
        #x = l.Bidirectional(l.LSTM(units=NUM_UNITS,return_sequences=True),merge_mode='ave')(x)
        
    else:
        x = l.LSTM(units=NUM_UNITS,return_sequences=True,input_shape=x.shape)(x)
        #x = l.LSTM(units=NUM_UNITS,return_sequences=True)(x)
                               
        #train_model.add(l.Flatten())

    #x = tf.keras.layers.Dense(100,activation='relu')(x)

    x = tf.keras.layers.Dense(1)(x)
    
    @tf.function
    def select_unit(t):
        one_hot, act = t[0],t[1]
        
        
        act = tf.tile(act,multiples=(1,1,(CANDIDATES+1)))
        
        
        act = act*one_hot
        act = tf.reduce_sum(act,axis=1)#tf.math.divide_no_nan(tf.reduce_sum(act,axis=1),tf.clip_by_value(tf.reduce_sum(one_hot,axis=1),1e-06,999.0) )
        
        act += 1e-7
        
        den =  tf.reduce_sum(act,axis=1)[:,None] #(BS,1)
        den = tf.tile(den,(1,(CANDIDATES+1))) #(BS,11)
        
        #act = tf.math.divide_no_nan(act,den)
        act = tf.math.softmax(act)
        return act   
    
    @tf.function
    def smooth_labels(labels, factor=0.01):
        # smooth the labels
        labels = tf.cast(labels,tf.float32)
        labels *= (1 - factor)
        labels += (factor / tf.cast(tf.shape(labels)[1],tf.float32))
        # returned the smoothed labels
        return labels
    
    x = l.Lambda(select_unit)([input_onehot,x])
    
    
    #x = l.Lambda(func)(x)
    @tf.function
    def eval(Y1,Y2):
        
        def compare_single_example(TUPLE):
            y1, y2 = TUPLE
            nz_ids=  (tf.greater(y1,0))
            num_nz_ids = tf.reduce_sum(tf.cast(nz_ids,tf.float32) )
            A1 = tf.where(nz_ids,1.0,0.0)
            
            args = tf.argsort(-y2)[0:tf.cast(num_nz_ids,tf.int32)]
            A2 = tf.reduce_sum(tf.one_hot(args,depth=tf.shape(y1)[0],dtype=tf.float32),axis=0)
            return (A1*A2/num_nz_ids, A1*A2/num_nz_ids)
        
        mapped_fn = tf.map_fn( compare_single_example,(Y1,Y2)   )[0]
        return tf.reduce_mean(tf.reduce_sum(mapped_fn,axis=-1))
        
        tf.print("Good:",tf.reduce_sum(tf.boolean_mask(Y2,tf.greater(Y1,0))),
                 "Bad:",tf.reduce_sum(tf.boolean_mask(Y2,tf.equal(Y1,0))))
        good = tf.reduce_sum(tf.boolean_mask(Y2,tf.greater(Y1,0)))
        bad = tf.reduce_sum(tf.boolean_mask(Y2,tf.equal(Y1,0)))
        return good / (good+bad)
    
    def xent(Y1,Y2):
        #tf.print("Labels:",Y1,summarize=-1)
        #tf.print("Pred:",Y2,summarize=-1)
        #tf.print("Error",tf.reduce_sum(tf.square(Y1-Y2)),summarize=-1)
        
        Y1 = tf.reshape(Y1,(-1,CANDIDATES+1))
        Y2 = tf.reshape(Y2,(-1,CANDIDATES+1))
        #tf.print("Labels2:",Y1)
        #tf.print("Pred2:",Y2)

        Y1 = smooth_labels(Y1)
        Y2 = smooth_labels(Y2)

        
        #return tf.reduce_mean(tf.square(Y1-Y2))
        return tf.reduce_mean(- tf.reduce_sum(Y1 * tf.math.log(Y2),axis=-1))

    
    train_model = keras.Model(inputs = [input_X],outputs=x)
    print(train_model.summary())
    
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.25*1e-2,
            decay_steps=100,
            decay_rate=0.98)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                   loss=xent, metrics=['acc'])
    
    
    
    OUT_TYPES =  ((tf.float32,tf.float32,tf.int32,tf.int32),tf.float32)
    OUT_SHAPES =  ((tf.TensorShape([None,SEQ_LEN,ATTR_SIZE+EMB_SIZE+(CANDIDATES+1)]),tf.TensorShape([None,(CANDIDATES+1)]),
                     tf.TensorShape([None,(CANDIDATES+1)]),tf.TensorShape([None,1])), 
                    tf.TensorShape([None,(CANDIDATES+1)]))
    
    
    def get_Gen(mode,batch_size):
        if mode == 'train':
            return tf.data.Dataset.from_generator( partial(meli_iterator,mode='train',batch_size=batch_size),output_types =OUT_TYPES,
                                               output_shapes =OUT_SHAPES)
        elif mode == 'val':
            return  tf.data.Dataset.from_generator( partial(meli_iterator,mode='val',batch_size=batch_size),output_types =OUT_TYPES,
                                               output_shapes =OUT_SHAPES)
        
    
    
    
    N_train = np.round(413163*0.8).astype(np.int32) + 1
    N_val = 413163 - N_train
    

    
    N_test = 177070
    
    train_steps = 1 + N_train // BATCH_SIZE 
    val_steps = 1 + N_val // BATCH_SIZE
    test_steps = 1 + N_test // BATCH_SIZE
    
    
    valGen = get_Gen('val',BATCH_SIZE)
    #testGen = trainGen = tf.data.Dataset.from_generator( partial(meli_iterator,'test'),output_types = (tf.float32,tf.float32),
    #                                           output_shapes = (tf.TensorShape([None,SEQ_LEN,17]),
    #                                                           tf.TensorShape([None,10]))
    #                                           )
    model_fname = path.join(DATA_DIR,'model',"RNN_MODEL.h5")
    #model.load_weights(path.join(DATA_DIR,"MY_MODEL_2.h5"))
    if not path.isfile(model_fname):
        input("Warning!!! Did not find model weights at {}. Training takes many hours! (Press ENTER)".format(model_fname))
        
        trainGen = get_Gen('train',BATCH_SIZE)
        DIV_NUM = 1
        train_model.fit(
        trainGen,
        steps_per_epoch=train_steps,
        epochs=2,
        shuffle=True,
        verbose=1)
        print("Saving weights...")
        train_model.save_weights(model_fname)
        print("Saving weights... Done!")

    else:
        train_model.load_weights(model_fname)
        print("Loaded weights...")



    BATCH_SIZE_2 = 2500
    train_steps = 1 + N_train // BATCH_SIZE_2
    val_steps = 1 + N_val // BATCH_SIZE_2
    test_steps = 1 + N_test // BATCH_SIZE_2

    def get_batches(mode, nbatch):
        print("Creating prediction dataframe...")
        L1 = []
        L2 = []
        i = 0
        gen = meli_iterator(mode,full=True,batch_size=BATCH_SIZE_2)
        for x,y in gen:
            print(np.round(i/nbatch,4))
            _,_, lks, actual = x
            
            pred = train_model.predict_on_batch(x)
            pred = np.array(pred)
            pred[lks < 0] = 0
            pred = np.round(pred,4)
            pred = pred[:,:CANDIDATES]
            lks = lks[:,:CANDIDATES]
            
            if i > nbatch:
                break
            L1.append(lks)
            L2.append(pred)
            i += 1
            
        df1 = pd.DataFrame(np.concatenate(L1,axis=0))
        df1.columms=['lk_{}'.format(i) for i in range(CANDIDATES)]
        df2 = pd.DataFrame(np.concatenate(L2,axis=0))
        df2.columns = ['pred_{}'.format(i) for i in range(CANDIDATES)]
        df = pd.concat([df1,df2],axis=1)
        
        if mode == 'train':
            df = df.iloc[0:N_train,:]
        elif mode == 'val':
            df = df.iloc[0:N_val,:]
        else:
            df = df.iloc[0:N_test,:]
        
        df.to_csv(path.join(DATA_DIR,'RNN_{}.csv'.format(mode)))
        

        
        
        print("Creating prediction dataframe...Done!")
        
    get_batches('val',val_steps)
    get_batches('test',test_steps)
    get_batches('train',train_steps)
    
    
    
if __name__ == '__main__':
    train_model()
