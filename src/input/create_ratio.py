'''
Created on 9 de nov de 2020

@author: klaus
'''
import jsonlines
from folders import DATA_DIR, SUBMISSIONS_DIR
import os
from os import path
import pandas as pd
import numpy as np
import urllib
import igraph as ig
from input.read_input import read_item_data,  get_emb


def create_ratio(mode = 'train',CUTOFF=50, which='domain_id',alternate=False):
    assert mode in ['train','val']
    assert which in ['domain_id','category_id','item_id','price','condition']
    df = read_item_data()
    
    df['price'] = pd.qcut(df['price'].values,100)

    dct_attr = df[which].to_dict()
    dct_dom = df['domain_id'].to_dict()
    

    if mode == 'train':
        check = lambda x: x <= np.round(413163*0.8).astype(np.int32)
    elif mode == 'val':
        check = lambda x: x > np.round(413163*0.8).astype(np.int32)
    else:
        raise Exception("mode must be train or val")
    
    DATA_PATH = path.join(DATA_DIR,'train_dataset.jl')
    i = 0 
    """ Create dictionary holding domain counts (searched, bought) """
    attr_s = dict([(k,0) for k in pd.unique(df[which])])
    attr_b = dict([(k,0) for k in pd.unique(df[which])])
    attr_o = dict([(k,0) for k in pd.unique(df[which])])
    with jsonlines.open(DATA_PATH) as reader:
        for obj in reader:
            if check(i):
                #print(i)
                L = []
                for h in obj['user_history']:
                    if h['event_type'] == 'view':
                        #print("Viewed {}".format(dct[h['event_info']]))
                        L.append(h['event_info'])
                    elif h['event_type'] == 'search':
                        #print("Searched {}".format(h['event_info']))
                        pass

                temp = pd.Series(L,index=range(len(L)),dtype=np.float64)
                
                L_k = pd.unique(L[::-1])[::-1]
                
                attr_unique = list(pd.unique([dct_attr[k] for k in L_k]))
                for dom in attr_unique:
                    if dom in attr_s:
                        attr_s[dom] += 1
                if alternate:
                    for attr in attr_unique:
                        if dct_dom[attr] == dct_dom[obj['item_bought']]:
                            attr_b[attr] += 1
                        else:
                            attr_o[attr] += 1
                else:
                    if dct_attr[obj['item_bought']] in attr_unique:
                        attr_b[dct_attr[obj['item_bought']]] += 1
                    else:
                        attr_o[dct_attr[obj['item_bought']]] += 1
                
            i += 1
            #L.append(obj)
            
    attr_b, attr_s = pd.DataFrame.from_dict(attr_b,orient = 'index'),\
                          pd.DataFrame.from_dict(attr_s,orient = 'index')
    attr_o = pd.DataFrame.from_dict(attr_o,orient = 'index')
                          
                          
                          
    attr_b.columns, attr_s.columns, attr_o.columns = ['bought'],['searched'], ['out_bought']
    attr_b['bought'] = attr_b['bought'].values.astype(np.float32)
    attr_s['searched'] = attr_s['searched'].values.astype(np.float32)
                       
    rat = attr_b['bought'].values/(1.0+attr_s['searched'].values)
    rat[attr_s['searched'].values < CUTOFF] = np.mean(rat[attr_s['searched'].values >= CUTOFF])

    rat2 = attr_o['out_bought'].values/(1.0+attr_b['bought'].values)
    rat2[attr_s['searched'].values < CUTOFF] = np.mean(rat2[attr_s['searched'].values >= CUTOFF])


    rat = pd.DataFrame({"rat":np.array(rat)},index=attr_b.index)
    rat2 = pd.DataFrame({"rat2":np.array(rat2)},index=attr_b.index)
    
    res = pd.concat([attr_s,attr_b,attr_o,rat,rat2],axis=1)
    if alternate:
        res.to_csv(path.join(DATA_DIR,'{}_ratio_alternate.csv'.format(which)))
    else:
        res.to_csv(path.join(DATA_DIR,'{}_ratio.csv'.format(which)))
    
    
def create_language():
    df = read_item_data()
    import fasttext
    model_fname = path.join(DATA_DIR,"lid.176.bin")
    if not path.isfile(model_fname):
        print("Did not find fasttext model at {}".format(model_fname))
        print("Trying to download from the web...")
        try:
            urllib.request.urlretrieve ("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", model_fname)

        except:
            raise Exception("Could not get fasttext model")
    if not path.isfile(model_fname):
        raise Exception("Could not get fasttext model")
    else:
        print("Fasttext model found at {}".format(model_fname))

        
    lid_model = fasttext.FastText.load_model(model_fname) 
    

    
    def get_language(i,x):
        print(i)
        languages, scores = lid_model.predict(str(x),k=999,threshold=-1.0)
        languages = np.array(languages)
        
        return scores[np.where(languages=='__label__es')[0][0]], scores[np.where(languages=='__label__pt')[0][0]], scores[np.where(languages=='__label__en')[0][0]]
    
    X = np.array([get_language(i,x) for i,x in enumerate(df['title'].values)])
    for i,c in enumerate(['score_es','score_pt','score_en']):
        df[c] = X[:,i]
    df.loc[:,['score_es','score_pt','score_en']].to_csv(path.join(DATA_DIR,'language_identification.csv'))
    
def load_language_df():
    return pd.read_csv(path.join(DATA_DIR,'language_identification.csv'),index_col=0)

def get_ratio(which='domain_id',full=False,standardize=False,alternate=False):
    assert which in ['domain_id','category_id','item_id','used']
    if alternate:
        fname = path.join(DATA_DIR,'{}_ratio_alternate.csv'.format(which))
    else:
        fname = path.join(DATA_DIR,'{}_ratio.csv'.format(which))
    df =  pd.read_csv(fname,index_col=0)
    if standardize:
        for c in df.columns:
            df[c] = (df[c] - np.mean(df[c].values))/np.std(df[c].values)
    if full:
        return df
    return df['rat'].to_dict()


def create_all():
    print("Creating language classification DataFrame...")
    create_language()
    print("Creating item_id feature DataFrame...")
    create_ratio('train',CUTOFF=0,which='item_id',alternate=False)
    print("Creating domain_id feature DataFrame...")
    create_ratio('train',CUTOFF=0,which='domain_id')
    print("Creating category_id feature DataFrame...")
    create_ratio('train',CUTOFF=0,which='category_id')  
    print("Creating (alternate) item id feature DataFrame...")
    create_ratio('train',CUTOFF=0,which='item_id',alternate=True)
    


if __name__ == "__main__":
    create_all()
    