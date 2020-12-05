# MercadoLibre_2020
Repo for my work on MercadoLibre Data Challenge 2020 

Required packages can be found in requirements.txt

<b>Main file to run is read_input.py</b>. It should check for the presence of a few files:

<b>"ratio files"</b>: These correspond to feature extraction. They are mostly ratios answering the question:<br><br>
<i>If a domain_id/item_id/category_id is made was viewed in the history object, then how likely is it that it is
the  domain_id/item_id/category_id  of the actual bought item???? </i><br>
  We also use a FastText model to extract features corresponding to the likelihood
of the item id belonging to the spanish/portuguese/english language, though it proved to be insignificant w.r.t. NDCG. To do this, we download a pre-trained fasttext model.

Notably, overfitting may occur for low-frequency items, so when training the LGB and RNN we recalculate the ratios 
so as if the current purchase and associated history were excluded from the training set, to avoid overfitting.

<b>"RNN_pred"</b>: We use a RNN model to predict the domain_id, by looking at the features of the last 30 unique items viewed. We also use the SentenceTransformer
package to extract a 512-dimensional embedding of the first two words.


<b>"Light Gradient Boosting (lgb.pkl)"</b>: Similar to the RNN, we train a LGBRanker to rank items. There is a special function that gathers the necessary input matrix with relevant features.

<b>"Neural_Domain_Identifier"</b>: A neural net that uses features from observed domains in the history object to predict the domain of the purchased object. Domains may repeat
several times with different items, so we extract max,min,mean,std. We also use the output of another model, which predicts normalized domain probabilities given a title string,
as extra features.

Our final model is hierarchical.

First, we rank the items. There are several rankings: recency,frequency,LGB model predictions, so on...

The LightGradientBoosting model (lgb.pkl) was trained to rank items. The RNN model also achieves the same thing.
We use hard-coded coefficients that performed well in the validation set. We also use "Neural_Domain_Identifier" to filter items whose domain receives a low score
(WARNING: it's better to use the saved model weights or the predictions already saved as csv. If you retrain, the optimal cutoff might have to be chosen again w.r.t. validation.

Finally, we use a directed graph to recommend items that we previously bought when viewing the same items as this purchase's. If that fails, we firstly rank by domain,
then recommend the most popular items ( 100*_times_bought + 1*_times_searched)





