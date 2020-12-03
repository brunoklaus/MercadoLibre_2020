'''
Created on 5 de nov de 2020

@author: klaus
'''

import jsonlines
from folders import DATA_DIR, SUBMISSIONS_DIR
import os
from os import path
import pandas as pd
import numpy as np

import igraph as ig



TRAIN_LINES = 413163
TEST_LINES = 177070
NUM_DOMS = 7894
import os, sys


def ndcg(pred,actual):
    """ Calculates the Normalized Discounted Cumulative Gain.
        
        pred (NDArray[N,10]): recommendations
        actual (NDArrau[N]): actual products
        
    """
    pred = pred.astype(np.int32)
    actual = actual.astype(np.int32)
    
    dct = read_item_data()['domain_id'].to_dict()
    
    def relevance(x,y,dom_x,dom_y):
        if x == y:
            return 12
        elif dom_x == dom_y:
            return 1
        else:
            return 0
        
    
    if pred.shape[1] != 10:
        raise ""
    N = pred.shape[0]
    
    L = np.zeros((N,))
    for i in range(N):
        actual_dcg = np.asarray(    [relevance(actual[i],pred[i,pos],
                                     dct[actual[i]],dct[pred[i,pos]]) for pos in range(10)]
                               )
        for i1, j in enumerate(np.where(actual_dcg == 12)[0]):
            if i1 > 0:
                actual_dcg[j] = 0
        ideal_dcg = np.asarray([12] + [1]*9)
        w = np.reciprocal(np.asarray([np.log2(2+pos) for pos in range(10)]))
        
        L[i] = np.sum(w*actual_dcg)/np.sum(w*ideal_dcg)
    
    return np.mean(L)
        


def create_item_graph(mode = 'train'):
    """
        Creates graph, whose vertices correspond to items. 
        For each purchase, an edge is added from each searched item to the one that was bought. 
        Edges may be repeated.
    """
    
    """
        Fetch data
    """
    TRAIN_LINES = 413163
    TEST_LINES = 177070
    df = read_item_data()
    df['item_id'] = df.index
    dct_title = df['title'].to_dict()
    dct_domain = df['domain_id'].to_dict()
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
    if mode == 'train':
        check = lambda x: x <= np.round(413163*0.8).astype(np.int32)
    elif mode == 'val':
        check = lambda x: x > np.round(413163*0.8).astype(np.int32)
    else:
        check = lambda x: True
    
    DATA_PATH = path.join(DATA_DIR,'test_dataset.jl' if mode == 'test' else 'train_dataset.jl')
    line_i = 0
    
    

    """
        Create graph vertices
    """
    g = ig.Graph() 
    
    counter, f_map_func, r_map_func = get_mappings()
    
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
                L = pd.unique(L)
                #L_domain = [dct_domain[k] for k in L]
                for i in range(len(L)):
                        E1.append(L[i])
                        E2.append(obj['item_bought'])
            
    
    
    E1 = f_map_func['item_id'](E1)
    E2 = f_map_func['item_id'](E2)
    
    
    E =  list(zip(E1,E2))
    g.add_edges(E)
    
    #g  = g.as_undirected()
                     
    g.write_pickle(fname=path.join(DATA_DIR,'graph_domain_id.pkl'))
    

def get_mappings():
    """
        For this project I hoped to create a graph that included items, domains and categories.
        Didn't have enough time to actually make good use of it, but this function serves to map
        an item/domain/category to its ID on the graph, and also from the ID back to it.
        
        The keys of the returned dictionaries are ``domain_id``,``category_id``,``item_id``
    
        
        Returns:
            counter (int) - number of vertices
            f_map_func (Dict[str,Dict[Union[str,int],int]]): maps from element to ID
            r_map_func (Dict[str,Dict[int,Union[str,int]]]): maps from ID to element
             
    """
    df = read_item_data()
    
    counter = 0
    f_map = {}
    r_map = {}
    for col in ['item_id','domain_id','category_id','product_id']:
        K = pd.unique(df[col].values)
        V = range(counter, counter + K.shape[0])
        counter += K.shape[0]
        print((col,counter))
        f_map[col] = dict(list(zip(K,V)))
        r_map[col] = dict(list(zip(V,K)))
    
    from functools import partial
    
    def forward(X,mp,k):
        mp_k = mp[k]
        return  [mp_k[x] for x in X]
    
    f_map_func = dict([(k,partial(forward,k=k,mp=f_map)) for k in f_map.keys()])
    r_map_func = dict([(k,partial(forward,k=k,mp=r_map)) for k in r_map.keys()])
    
    return counter, f_map_func, r_map_func
    

#SENTENCE_MODEL_CHOICE = 'distilbert-base-nli-mean-tokens'
SENTENCE_MODEL_CHOICE = 'distiluse-base-multilingual-cased-v2'
emb_shape = 512
__sentence_model = None
def get_sentence_model():
    """
        Returns a model from the SentenceTransformer package,
        which is used to transform strings into embeddings.
    """
    global __sentence_model
    if __sentence_model is None:
        print("Loading TRANSFORMER MODEL.... ")
        from sentence_transformers import SentenceTransformer
        __sentence_model = SentenceTransformer(SENTENCE_MODEL_CHOICE)
    return __sentence_model

from scipy  import spatial
def cos_dist(x,y):
    return spatial.distance.cosine(x,y)
"""
    saved_emb is a dictionary that serves as a lookup so that we don't calculate the same embeddings over and over.
"""
saved_emb = {}

def get_emb_Kstr(X,K):
    """
        Get embedding for a given list of strings, using a list of strings as the key for lookup.
        
        Args:
            X (List[str]): list of input strings for which we want an embedding. (Can be words or sentences)
            K (List[str]): list of corresponding keys
        Returns:
            NDArray[N,512]: embeddings
        
    """
    K = np.array(K)
    X = np.array(X)
    b = np.array([k in saved_emb.keys() for k in K])
    b_where = np.where(b)[0]
    notb_where = np.where(np.logical_not(b))[0]
    res = np.zeros((len(X),emb_shape))
    lookup = [saved_emb[x] for x in K[b_where]]
    if len(lookup) > 0:
        res[b_where,:] = np.concatenate([saved_emb[x] for x in K[b_where]],axis=0)
    if len(notb_where) > 0:
        model_out = get_sentence_model().encode([k.lower() for k in X[notb_where]])
        for i,b  in enumerate(notb_where):
            if (isinstance(K[b],str) or K[b] >= 0) and len(saved_emb) < 1000000:
                saved_emb[K[b]] = model_out[i,:][None,:]
        res[notb_where,:] =  model_out

    return res

def get_emb(X,K):
    """
        Get embedding for a given list of strings, using a list of integers as the key for lookup.
        
        Args:
            X (List[str]): list of input strings for which we want an embedding. (Can be words or sentences)
            K (List[int]): list of corresponding keys
        Returns:
            NDArray[N,512]: embeddings
        
    """
    X = np.array(X)
    K = np.array(K).astype(np.int32)
    b = np.array([k in saved_emb.keys() for k in K])
    b_where = np.where(b)[0]
    notb_where = np.where(np.logical_not(b))[0]
    res = np.zeros((len(X),emb_shape))
    lookup = [saved_emb[x] for x in K[b_where]]
    if len(lookup) > 0:
        res[b_where,:] = np.concatenate([saved_emb[x] for x in K[b_where]],axis=0)
    if len(notb_where) > 0:
        model_out = get_sentence_model().encode([k.lower() for k in X[notb_where]])
        for i in range(model_out.shape[0]):
            if (isinstance(K[i],str) or K[i] >= 0) and len(saved_emb) < 1000000:
                saved_emb[K[i]] = model_out[i,:][None,:]
        res[notb_where,:] =  model_out
        
    return res

def first_two_words(x):
    return ' '.join([k for k in x.split(' ') if not any([i.isdigit() for i in k]) ][:2])

def get_lgb_data(avoid_overfit=True):
    """
        Gets all the features necessary to train the LGB ranker, arranging them into matrices.
        Args:
            avoid_overfit (bool): If ``True``, avoid overfitting by decreasing the item/domain/category bought/searched count
            for the elements from the history of a given purchase. Default is ``True``.
        
        Returns: List with size 3:
            X (NDArray[float].shape[N,D]): Features
            Y (NDArray[float].shape[N,1]): Labels
            M (NDArray[float].shape[N]): Indicator variable (1 if train, 0 if validation)
            
        
    """
    from input.create_ratio import load_language_df
    mode = 'train'
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
    
        #print(df2)

    """ 
        RNN stuff.
    """
    from input.rnn_item_ranker import read_predictions
    rnn_pred = read_predictions(mode)
    #assert rnn_pred.shape[0] == TRAIN_LINES
    

    
    DATA_PATH = path.join(DATA_DIR,'test_dataset.jl' if mode == 'test' else 'train_dataset.jl')
    i = 0
    
    
    def rank_to_order(L,rank):
        assert rank.shape[0] == L.shape[0]
        ids = (-rank).argsort(kind='mergesort')
        return L[ids], rank[ids] 
    
    pred = {}
    
    actual = []
    domain_ids = []
    
    
    X = []
    Y = []
    M = []
    with jsonlines.open(DATA_PATH) as reader:
        for line_id, obj in enumerate(reader):
            if True:
                print(line_id)
                L = [h['event_info'] for h in obj['user_history'] if h['event_type'] == 'view']
                S =  [h['event_info'] for h in obj['user_history'] if h['event_type'] == 'search']
                
                L_k = pd.unique(L[::-1])[::-1]
                
                """
                    OVERFITTING AVOIDANCE
                
                """
                if avoid_overfit:
                     
                    if line_id <= 330530:
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
                            
                            
                            
                            
                            
                                

                            
                            
                            
                            
                            
                            
                            
                            
                   
                
                
                
                           
                """
                    Calculate ranks
                """
                
                dct_rnn = dict([(int(x),y) for x,y in zip(rnn_pred.iloc[i,0:10],rnn_pred.iloc[i,-10:])])
                if len(L_k) <= 10:
                    rank_ratio_rnn = pd.Series([dct_rnn.get(k,0) for k in L_k]).rank(method="average").to_numpy()
                else:
                    rank_ratio_rnn = pd.Series([1.0 for k in L_k]).rank(method="average").to_numpy()
                    
                rank_ratio_dom = pd.Series([dct_ratio_dom[dct_domain[k]] for k in L_k]).rank(method="average").to_numpy()
                rank_ratio_cat = pd.Series([dct_ratio_cat[dct_cat[k]] for k in L_k]).rank(method="average").to_numpy()
                rank_ratio_item = pd.Series([dct_ratio_item_p[k] for k in L_k]).rank(method="average").to_numpy()
                
                rank_freq = pd.Series(L,index=range(len(L))).value_counts(sort=False) 
                rank_freq = rank_freq.rank(method="average").to_dict()
                rank_freq = np.array([rank_freq[k] for k in L_k])
                rank_latest = np.arange(len(L_k))
                rank_price = pd.Series([-dct_price[k] for k in L_k]).rank(method="average").to_numpy()
                    
                

                x = []
                x.append([dct_ratio_dom[dct_domain[k]] for k in L_k])
                x.append(rank_ratio_dom)
                x.append(rank_ratio_cat)
                x.append(rank_price)
                
                         
                            
                
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
                    Overfitting avoidance - pt 2
                """
                if line_id <= 330530:
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
                            
                
                if len(L_k) == 0: 
                    continue
                x = np.transpose(np.reshape(np.array(x),(-1,len(L_k))) )

                def score(k):
                    if k == obj['item_bought']:
                        return 2
                    elif dct_domain[k] == dct_domain[obj['item_bought']]:
                        return 1
                    else:
                        return 0
                    
                y = np.array([ score(k) for k in L_k])[:,None] 
                #print(y.shape)
                if np.sum(y) >= 0:
                    X.append(x)
                    Y.append(y)
                    M.append(np.array([line_id]*len(L_k)))
                

                
    

    
    X = np.concatenate(X,axis=0)
    Y = np.concatenate(Y,axis=0)
    M = np.concatenate(M)
    return X,Y,M
        

def lgb_rank():
    """
        Trains the LGB Ranker and saves it to the data directory.

    """
    #print(0.8*TRAIN_LINES)
    
    import lightgbm as lgb
    import numpy as np
    params = {
    'num_iterations':2000,
    'learning_rate': 0.0025,
    'bagging_fraction': 0.25,
    'feature_fraction':0.8,
    'bagging_freq':1,
    "boosting":'gbdt',
    'feature_fraction':1,
    'early_stopping_round':100,
    'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'max_depth': 3,  # -1 means no limit
    }
    X,Y, M = get_lgb_data(avoid_overfit=True)
    
    lgbc = lgb.LGBMRanker(**params)
    
    X_train, X_val = X[M <= 330530,:], X[M > 330530,:]
    Y_train, Y_val = np.ravel(Y[M <= 330530,:]), np.ravel(Y[M > 330530,:])
    
    M_train, M_val = M[M <= 330530], M[M > 330530]
    
    def get_successive_sizes(M):
        M = np.array(M)
        u, unique_ids = np.unique(M,return_index=True)
        unique_ids = np.sort(unique_ids)
        unique_ids = list(unique_ids)
        unique_ids.append(M.shape[0])
        for i in range(len(unique_ids)-1):
            unique_ids[i] = unique_ids[i+1] - unique_ids[i]
        unique_ids.pop()
        return unique_ids
    
    M_train, M_val = [np.asarray(get_successive_sizes(x)) for x in [M_train,M_val]]
    
    
    bst = lgbc.fit(X_train,Y_train,group=M_train,eval_group=[M_val],
               eval_set=[(X_val,Y_val)],verbose=2)
    
    pred_train = bst.predict(X_train,group=M_train)
    pred_val = bst.predict(X_val,group=M_val)

    from sklearn.externals import joblib
    # save model
    joblib.dump(lgbc, path.join(DATA_DIR,'model','lgb.pkl'))
    
        

import scipy.spatial


def final_prediction(mode = 'train',use_graph=True,debug=False):
        """
            Combines all classifiers in a hierarchical manner to create the final predictions.
            
                First, we create many rankings for items seen during the object history, such as ones based on frequency and recency.
            Perhaps the most important rankings are the ones related to the predictions of the RNN and LGB. I have hardcoded some
            coefficients that attained good validation accuracy. The top 10 items are selected.
            
                Then, I use the Neural Domain Classifier's predictions to eliminate items among those 10, specifically ones whose domain
                is very unlikely to be the one. Once again, there is a hardcoded cutoff that may need some tuning if you train the classifier from
                scratch, as it can have a significant effect on the NDCG.

        """
        TRAIN_LINES = 413163
        TEST_LINES = 177070
        df = read_item_data()

        from input.create_ratio import load_language_df
        df2 = load_language_df()
        dct_lan_pt = df2['score_pt'].to_dict()
        dct_lan_en = df2['score_en'].to_dict()
        dct_lan_es = df2['score_es'].to_dict()
        
        dct_condition = df['condition'].to_dict()

        dct = df['title'].to_dict()
        dct_domain = df['domain_id'].to_dict()
        dct_cat = df['category_id'].to_dict()
        dct_price = df['price'].to_dict()
        dct_pid = df['product_id'].to_dict()
    
        """ Ratio stuff """    
        from input.create_ratio import get_ratio
        dct_ratio_dom = get_ratio(which='domain_id')
        
        ratio_df = get_ratio(which='item_id',full=True)
        ratio_df['popularity'] = 100.0*ratio_df['bought'] + ratio_df['searched']
        dct_ratio_item_p = ratio_df['popularity'].to_dict()
        
        
        
        """ Most common embeddings. """
        ratio_df = ratio_df.sort_values(['popularity'],ascending=False)
        most_common_emb = get_emb([first_two_words(dct[k]) for k in ratio_df.index[0:100]],[-1]*100)
        
        ratio_df = get_ratio(which='item_id',full=True,alternate=False)
        dct_ratio_item_b = ratio_df['bought'].to_dict()
        dct_ratio_item_s = ratio_df['searched'].to_dict()
        dct_ratio_item_r = ratio_df['rat'].to_dict()
        
        df['item_popularity'] = [dct_ratio_item_p[k] for k in df.index]
        
        dct_ratio_cat = get_ratio(which='category_id')
        dct_ratio_item = get_ratio(which='item_id')
        
        
        
        dct_domain_df = {}
        dct_cat_df = {}
        for dom, df2 in df.groupby('domain_id'):
            df2= df2.sort_values(['item_popularity'],ascending=False)#.iloc[0:10,:]
            dct_domain_df[dom] =  df2
        
        for cat, df2 in df.groupby('category_id'):
            df2 = df2.sort_values(['item_popularity'],ascending=False)#.iloc[0:10,:]
            dct_cat_df[cat] =  df2
        
            #print(df2)
            

        """ 
            RNN stuff.
        """
        from input.rnn_item_ranker import SEQ_LEN,CANDIDATES
        from input.rnn_item_ranker import read_predictions
        rnn_pred = read_predictions(mode)
        assert rnn_pred.shape[1] == 2*CANDIDATES
        if mode == 'train' or mode == 'val':
            assert rnn_pred.shape[0] == TRAIN_LINES
        
    
    
        """
            LGB stuff
        """
    
        import lightgbm as lgb

        from sklearn.externals import joblib
        lgbc = joblib.load(path.join(DATA_DIR,'model','lgb.pkl'))
        """
            Graph-related initialization
        """
        graph_fname = path.join(DATA_DIR,'graph_domain_id.pkl')
        if not path.isfile(graph_fname):
            print("Creating item-to-item graph")
            create_item_graph(mode='train')
        G1 = ig.Graph.Read_Pickle(graph_fname)
        _, f_map_func, r_map_func = get_mappings()
        
        if mode == 'test':
            DF_DOM_PRED = pd.read_csv(path.join(DATA_DIR,'domain_pred_test.csv'),index_col=0)

        else:
            DF_DOM_PRED = pd.concat([pd.read_csv(path.join(DATA_DIR,'domain_pred_train.csv'),index_col=0),
                                 pd.read_csv(path.join(DATA_DIR,'domain_pred_val.csv'),index_col=0)],
                                 ignore_index=True)
        DF_CONF_PRED = DF_DOM_PRED.loc[:,['conf_{}'.format(i) for i in range(10)[::-1]]] 

        DF_DOM_PRED = DF_DOM_PRED.loc[:,['pred_{}'.format(i) for i in range(10)[::-1]]] 
        vals = pd.unique(df['domain_id'].values)
        for c in DF_DOM_PRED.columns:
            DF_DOM_PRED[c] = DF_DOM_PRED[c].values.astype(np.int32)
            DF_DOM_PRED[c] = [vals[k] for k in DF_DOM_PRED[c]]
            
  
        
        """
            EMB stuff
        """
        from gcn.domain_string_identifier import predict_model, load_model
        domain_identifier = load_model()
        
        if mode == 'train':
            check = lambda x: x <= np.round(413163*0.8).astype(np.int32)
        elif mode == 'val':
            check = lambda x: x > np.round(413163*0.8).astype(np.int32)
        else:
            check = lambda x: True
        
        DATA_PATH = path.join(DATA_DIR,'test_dataset.jl' if mode == 'test' else 'train_dataset.jl')
        i = 0
        
        
        def rank_to_order(L,rank):
            assert rank.shape[0] == L.shape[0]
            ids = (-rank).argsort(kind='mergesort')
            return L[ids], rank[ids] 
        
        pred = {}
        res = []
        actual = []
        domain_ids = []
        lgb_acc = 0
        rnn_acc = 0
        counter = 0
        del df
        del df2
        #_scores = np.zeros((10,)).astype(np.float32)
        with jsonlines.open(DATA_PATH) as reader:
            for line_id, obj in enumerate(reader):
                
                
                def score(k):
                    if k == obj['item_bought']:
                        return 12
                    elif dct_domain[k] == dct_domain[obj['item_bought']]:
                        return 1
                    else:
                        return 0
                
                if check(line_id):
                    
                    print("Current line {}".format(line_id))
                    L = [h['event_info'] for h in obj['user_history'] if h['event_type'] == 'view']
                    S =  [h['event_info'] for h in obj['user_history'] if h['event_type'] == 'search']
                  
                  
                    L_k = pd.unique(L[::-1])[::-1]
 
                    """
                        Calculate ranks
                    """
                    
                    if len(L_k) > 0:    
                        
                        rank_ratio_dom = pd.Series([dct_ratio_dom[dct_domain[k]] for k in L_k]).rank(method="average").to_numpy()
                        rank_ratio_cat = pd.Series([dct_ratio_cat[dct_cat[k]] for k in L_k]).rank(method="average").to_numpy()
                        rank_ratio_item = pd.Series([dct_ratio_item_p[k] for k in L_k]).rank(method="average").to_numpy()
                        
                        rank_freq = pd.Series(L,index=range(len(L))).value_counts(sort=False) 
                        rank_freq = rank_freq.rank(method="average").to_dict()
                        rank_freq = np.array([rank_freq[k] for k in L_k])
                        rank_latest = np.arange(len(L_k))
                        rank_price = pd.Series([-dct_price[k] for k in L_k]).rank(method="average").to_numpy()
                        
                        
                        vals = DF_DOM_PRED.iloc[line_id,:].values
                        RANK_DOM = [np.where(vals == dct_domain[k])[0] for k in L_k]
                        RANK_DOM = [vals.shape[0] - k[0] if len(k) > 0 else 0  for k in RANK_DOM]
                        RANK_DOM = pd.Series(RANK_DOM).rank(method="average").to_numpy()
                        


                        from input.rnn_item_ranker import SEQ_LEN,CANDIDATES
                        dct_rnn = dict([(int(x),y) for x,y in zip(rnn_pred.iloc[line_id,0:CANDIDATES],rnn_pred.iloc[line_id,-CANDIDATES:])])
                        
                        if len(L_k) <= CANDIDATES:
                            try:
                                rank_ratio_rnn = pd.Series([dct_rnn[k] for k in L_k]).rank(method="average").to_numpy()
                            except:
                                
                                print(L_k)
                                print(rnn_pred.iloc[(line_id-5):(line_id+10),:])
                                raise ValueError("Did not find keys in RNN prediction")
                                raise ""
                        else:
                            rank_ratio_rnn = pd.Series([1.0 for k in L_k]).rank(method="average").to_numpy()
                        
    
                        
                        
                        x = []
                        x.append([dct_ratio_dom[dct_domain[k]] for k in L_k])
                        x.append(rank_ratio_dom)
                        x.append(rank_ratio_cat)
                        x.append(rank_price)
                        
                        
                        x.append([dct_ratio_item_b[k] for k in L_k])
                        
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
                        
                        x = np.transpose(np.reshape(np.array(x),(-1,len(L_k))) )
                        
                        rank_lgb  = pd.Series(lgbc.predict(x)).rank(method="average").to_numpy()
                        
                        if (not mode == 'test') and obj['item_bought'] in L_k and len(L_k) <= CANDIDATES:
                            if L_k[np.argmax(rank_lgb)] == obj['item_bought']:
                                lgb_acc += 1
                            if L_k[np.argmax(rank_ratio_rnn)] == obj['item_bought']:
                                rnn_acc += 1
                            counter += 1

                        
                        COEFFS = [1.5,1.5,4.5,0.4,0.4,0.6,0.8,0.0]
                        COEFFS = np.array(COEFFS)/np.sum(COEFFS)
                        final_rank =       COEFFS[0]*rank_freq + \
                                           COEFFS[1]*(rank_lgb) + \
                                           COEFFS[2]*(rank_ratio_rnn) + \
                                           COEFFS[3]*(rank_ratio_dom) +\
                                           COEFFS[4]*(rank_ratio_cat)+\
                                           COEFFS[5]*(rank_ratio_item)+\
                                           COEFFS[6]*(rank_latest)+\
                                           COEFFS[7]*(rank_price)
                        
                        
                        """
                            Yield rank
                        """
                        L, L_ranks = rank_to_order(L_k, final_rank)
                        #L = L[rank_freq.argsort(kind='mergesort')]
         
                        #L = np.array([d for d in L if (dct_ratio_dom[dct_domain[d]] > 0.01 and dct_ratio_cat[dct_cat[d]] > 0.01)])
                        #DF_DOM_PRED.iloc[line_id,:] = DF_DOM_PRED.iloc[line_id,:]/np.max(DF_DOM_PRED.iloc[line_id,:])
                        #print(DF_CONF_PRED.iloc[line_id,:])
                        #print(DF_CONF_PRED.iloc[line_id,:] > 0.001)
                        #print(np.where(DF_CONF_PRED.iloc[line_id,:] > 0.001)[0])
                        
                        b = np.where(DF_CONF_PRED.iloc[line_id,:] > 0)[0]
                        vals = DF_DOM_PRED.iloc[line_id,:].values[b]
                        L = np.array([k for k in L if dct_domain[k] in vals])
                        
                        L = np.array([k for k in L if dct_rnn.get(k,1) > 1e-02])
                        L = L[:10]
                        

                        
                        
                        P = np.zeros((10,),dtype=np.int32)
                        P[0:L.shape[0]] =  L
                    else:
                        P = np.zeros((10,),dtype=np.int32)
                        L = np.array(L)
                    
                    

                     
                     
                    
                   
    
                    TEMP_MAX = 101
                    if len(obj['user_history']) > 0:
                        temp = []
                        doms = [dct_domain[k] for k in L]
                        if len(L) > 0:
                            score_en = np.nanmean([dct_lan_en[k] for k in L])
                            score_es = np.nanmean([dct_lan_es[k] for k in L])
                            score_pt = np.nanmean([dct_lan_pt[k] for k in L])
                        else:
                            score_en,score_es,score_pt=0,0,0
                        
                        b = np.where(DF_CONF_PRED.iloc[line_id,:] > 1e-05)[0]
                        doms = DF_DOM_PRED.iloc[line_id,:].values[b]
                        
                        cats = [x[1] for x in sorted([(-dct_ratio_cat[k],str(k)) for k in [dct_cat[k] for k in L]])]
                        cat_rating = dict([(k,-dct_ratio_cat[k]) for k in cats])
                        
                        
                        if use_graph:
                            roots = pd.unique([k for k in L])
                            roots = f_map_func['item_id'](roots)
                        for dom in doms:
                            
                            if use_graph and len(roots) > 0:
                                c_score = {}
                                candidates = []
                                for k in roots:
                                    source_vert = G1.vs[k]
                                    es = G1.incident(source_vert,mode='OUT')
                                    es = G1.es[es]
                                    vs = [e.target for e in es]
                                    
                                    N = len(vs)
                                    vs = G1.vs[vs].select(domain_id=dom)
                                    vs = [v['value'] for v in vs]
                                    candidates.extend(vs)
                                if len(candidates) > 0:
                                    candidates = pd.Series([k for k in candidates]).value_counts()
                                    candidates = candidates[candidates.values > 1]
                                    _temp = [k for k in list(candidates.index) if not k in temp]
                                    temp.extend(_temp)
                                    
                                                             
                            
                            if dom in dct_domain_df.keys():
                                if len(temp) > 40:
                                    break
                                x = dct_domain_df[dom].index[0:TEMP_MAX]
                                
                                """
                                    Here we try to restrict to items in the same language. This had minimal effect on the NDCG.
                                """
                                if score_pt -  score_es > 0.4:
                                    x = [k for k in x if score_pt - dct_lan_pt[k] < 0.2]
                                elif score_es -  score_pt > 0.4:
                                    [k for k in x if score_es - dct_lan_es[k] < 0.2]
                                    
                                
                                x = sorted(x, key=lambda k: cat_rating[dct_cat[k]] if dct_cat[k] in cats else 0 )
                                temp.extend(x)
                            
                        ##############################################################    
                        """ Add more items if there aren't enough"""
                        temp = temp[0:TEMP_MAX]
                        temp = [k for k in temp if k not in L]
                        
                        x = 0
                        while len(pd.unique(temp)) < 10:
                            if isinstance(DF_DOM_PRED.iloc[line_id,x],str):
                                temp.extend(dct_domain_df[DF_DOM_PRED.iloc[line_id,x]].index[0:10])
                            x += 1
                              
                        temp = [k for k in temp if k not in L]
                            
                        temp = pd.unique(temp)
                        
                            
                        ########################################################3
                        """ Finally, add the ranked items to our prediction. """
                        P[L.shape[0]:] = temp[:(10-L.shape[0])]

                        
                    else:
                        """ Special case for empty search and item"""
                        x = 0
                        while len(pd.unique(temp)) < 10:
                            if isinstance(DF_DOM_PRED.iloc[line_id,x],str):
                                temp.extend(dct_domain_df[DF_DOM_PRED.iloc[line_id,x]].index[0:10])
                            x += 1
                              
                        temp = [k for k in temp if k not in L]
                            
                        temp = pd.unique(temp)
                    
                    """
                        Set prediction
                    """
                    pred[line_id] = P
                    
                    actual.append(obj.get('item_bought',0))
                    if len(actual) > 10000 and debug:
                        #print(lgb_acc/counter,rnn_acc/counter)
                        break
                    
                    

                    #print("Item bought: {}".format(dct[obj['item_bought']]))
                #L.append(obj)
        
        """
            Now we calculate NDCG and save our prediction DataFrame.
        """
        if mode == 'test':
            pred = np.reshape(np.asarray(list(pred.values()) ),(-1,10))
            OUT_PATH = path.join(SUBMISSIONS_DIR,'submission.csv')
            out_df = pd.DataFrame(data=pred,index=range(pred.shape[0]),columns=range(pred.shape[1]))
            out_df.to_csv(OUT_PATH,index=False,header=False)
        else:
            pred = np.reshape(np.asarray(list(pred.values()) ),(-1,10))
            print(pred)
            actual = np.asarray(actual)
            res = ndcg(pred,actual)
            print("Number of objects: {}".format(pred.shape[0]))
            print(COEFFS)
            print("NDCG: {}".format(res))
            return -res
            
import warnings

def read_item_data():
    """
        Creates or loads item data DataFrame from item_data.jl. The dataframe is as ``item_data.csv`` in the data folder.
    """
    cols = ['title',
            'price',
            'category_id',
            'category_path_from_root',
            'catalog_product_id',
            'domain_id'
            ]
    L = []
    ITEM_PATH = path.join(DATA_DIR,'item_data.jl')
    ITEM_CSV_PATH = path.join(DATA_DIR,'item_data.csv')
    
    if not path.isfile(ITEM_CSV_PATH):
        with jsonlines.open(ITEM_PATH) as reader:
            for obj in reader:
                L.append(obj)
        df = pd.DataFrame.from_dict(L)
        """ Treat NA.
        
        title -- # of NA entries: 0.00 
        domain_id -- # of NA entries: 851.00 
        product_id -- # of NA entries: 1805749.00 
        price -- # of NA entries: 339.00 
        category_id -- # of NA entries: 0.00 
        condition -- # of NA entries: 856.00 
        """
        df['product_id'] = df['product_id'].fillna(0)
        df['price'] = df['price'].fillna(-100)
        df['condition'] = df['condition'].fillna("unknown")
        df['domain_id'] = df['domain_id'].fillna("")

        
        
        df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')
        df.sort_values(by=['item_id'])

        df = df.set_index('item_id')

        df.to_csv(ITEM_CSV_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)  
        df = pd.read_csv(ITEM_CSV_PATH,index_col=0)
        
    df['item_id'] = list(df.index)
    df['condition'] = [0 if k == 'new' else 1 for k in df['condition'].values]
    return df

def question(_str):
    answer = ''
    while not answer in ['y','n']:
        answer  = input(_str).lower()
        if not answer in ['y','n']:
            print("Please answer with Y (yes) or N (no) ")
    return answer == 'y'
     

def main():
    for fname in ['item_data.jl','test_dataset.jl','train_dataset.jl']:
        _link = "https://meli-data-challenge.s3.amazonaws.com/2020/{}.gz".format(fname)
        _out_path = path.join(DATA_DIR,"{}.gz".format(fname))
        _out_path_jl = path.join(DATA_DIR,"{}".format(fname))
        import urllib.request as request
        if not path.isfile(_out_path):
            print("Downloading {} from {}".format(fname,_link))
            try:
                request.urlretrieve(_link,_out_path)
                assert path.isfile(_out_path)
            except:
                raise "Could not get {} from {} or file not found at {}".format(fname,_link,_out_path)
            
            
        if not path.isfile(_out_path_jl):
            import gzip
            import shutil
            print("Extracting {} as {}...".format(_out_path,_out_path_jl))
            def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
                with gzip.open(source_filepath, 'rb') as s_file, \
                        open(dest_filepath, 'wb') as d_file:
                    shutil.copyfileobj(s_file, d_file, block_size)
            
            gunzip_shutil(_out_path, _out_path_jl)
            print("Extracting {} as {}... Done!".format(_out_path,_out_path_jl))
    
    
    
    LGB_MODEL_PATH = path.join(DATA_DIR,'model','lgb.pkl')
    RATIO_PATH = path.join(DATA_DIR,'domain_id_ratio.csv')
    
    NEURAL_DOMCLASSIFIER_PATH = path.join(DATA_DIR,'domain_pred_test.csv')
    RNN_PATH = path.join(DATA_DIR,'RNN_test.csv')
    
    
    """
        Feature extraction
    """
    if path.isfile(RATIO_PATH):
        print("Detected features at path {} ".format(RATIO_PATH))
        Q = question('Extract features from scratch anyway? (Y/N)')
    else:
        print("Did NOT features at path {} ".format(RATIO_PATH))
        Q = question("Extract features from scratch? (Y to confirm)")
    if Q:
        from input.create_ratio import  create_all
        create_all()
    
    """
        LGB item ranker
    """
    if path.isfile(LGB_MODEL_PATH):
        print("Detected pre-trained LGB classifier at path {} ".format(LGB_MODEL_PATH))
        Q = question('Train LGB from scratch anyway? (Y/N)')
    else:
        print("Did NOT detect pre-trained LGB classifier at path {} ".format(LGB_MODEL_PATH))
        Q = question("Train LGB from scratch? (Y to confirm)")
    if Q:
        lgb_rank()
        
    """
        RNN item ranker
    """
    if path.isfile(RNN_PATH):
        print("Detected pre-trained RNN item ranker at path {} ".format(RNN_PATH))
        Q = question('Train RNN item ranker from scratch anyway? (Y/N)')
    else:
        print("Did NOT detect pre-trained RNN item ranker at path {} ".format(RNN_PATH))
        Q = question("Train RNN item ranker from scratch? (Y to confirm)")
    if Q:
        from input.rnn_item_ranker import train_model
        train_model()


    """
        Neural Net Domain classifier
    """
    if path.isfile(NEURAL_DOMCLASSIFIER_PATH):
        print("Detected pre-trained NEURAL DOMAIN CLASSIFIER at path {} ".format(NEURAL_DOMCLASSIFIER_PATH))
        Q = question('Train NEURAL DOMAIN CLASSIFIER from scratch anyway? (Y/N)')
    else:
        print("Did NOT detect pre-trained NEURAL DOMAIN CLASSIFIER at path {} ".format(NEURAL_DOMCLASSIFIER_PATH))
        Q = question("Train NEURAL DOMAIN CLASSIFIER from scratch? (Y to confirm)")
    if Q:
        from gcn.predict_dom_v2 import train_neural_domain_prediction
        train_neural_domain_prediction()
    
    m = ''
    while not m in ['val','train','test']:
        m = input("Create predictions for which dataset?? (val/train/test)").lower().strip()
    final_prediction(mode=m,use_graph=True,debug=True)
    


if __name__ == "__main__":
    main()
    