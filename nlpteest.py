# # !pip install nltk
# # !pip install emoji
# # !pip install autocorrec\t
# # !pip install xgboost
# # !pip install -U spacy
# # nltk.download('all-corpora')
# # nltk.download('punkt')  # for using word_tokenizer
# # nltk.download('wordnet')  # for using Lemmatizer
# # nltk.download('averaged_perceptron_tagger') # for language processing i.e tagging words with their parts of speech (POS)
import subprocess
# subprocess.call('python3 -m spacy download en_core_web_lg')

from sklearn.preprocessing import MaxAbsScaler

import os
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import spacy
import nltk
import re  # regular expression
import string
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

import scipy.sparse as sp
import joblib
import emoji
from autocorrect import Speller  # for correcting spelling
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  # for tokenizing string into words
from nltk.stem import WordNetLemmatizer  # for lemmatizing words
from nltk.tag import pos_tag  # for tagging words with their parts of speech (POS)
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from gensim.models import Phrases
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import xgboost
# from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score


###############################################


def do_preprocess():
    # english Stopwords
    stopwords_list = stopwords.words('english')
    # Emoticons
    with open('EMOTICONS.pkl', 'rb') as f1:
        EMOTICONS = pickle.load(f1)
    # Contractions
    with open('CONTRACTIONS.pkl', 'rb') as f2:
        CONTRACTIONS = pickle.load(f2)

    def preprocess_text(tweet):

        nan_tweet = 'OOV'
        # this code is to short unnecessary sentence, bec. some rows has unnecessary long repeated characters
        # like 'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH'
        # we manually decide len = 10 any with len >10 is discarded
        # convert all text lowercase
        tweet = tweet.lower()
        tweet = tweet.split()
        tw = []
        for t in tweet:
            # removing digits only
            if t.isnumeric():
                continue
            else:
                pass
            if len(t) <= 20:
                if len(set(t)) <= 1:
                    continue
                if sum(c.isdigit() for c in t) > sum(c.isalpha() for c in t):
                    continue
                tw.append(t)
        tweet = ' '.join(tw)
        # remove any urls
        tweet = re.sub(r"www\S+|http\S+|", "", tweet, flags=re.MULTILINE)
        # remove square bracket including its content if
        tweet = re.sub(r'\[|\]', " ", tweet)
        # to remove new line character
        tweet = re.sub(r'\n', " ", tweet)
        # remove user @ references and '#' from tweet
        tweet = re.sub(r"\@\w+|\#", "", tweet)
        # replace emojis with its meaning
        tweet = (emoji.demojize(tweet, delimiters=(" ", ""))).replace('_', ' ')
        # expand contractions
        splitted_string = tweet.split()
        for index, text in enumerate(splitted_string):
            if text in CONTRACTIONS.keys():
                splitted_string[index] = CONTRACTIONS[text]
        tweet = ' '.join(splitted_string)
        # replace emoticons with its meaning
        splitted_tweet = tweet.split()
        for index, word in enumerate(splitted_tweet):
            if word in EMOTICONS.keys():
                splitted_tweet[index] = EMOTICONS[word]
        tweet = ' '.join(splitted_tweet)
        # remove tags
        tweet = re.sub(re.compile('<.*?>'), '', tweet)
        # remove punctuations
        tweet = tweet.translate(str.maketrans("", "", string.punctuation))
        # remove stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [word for word in tweet_tokens if word not in stopwords_list]
        # spelling correction
        correct_words = []
        # initialize Speller object for english language
        spell_corrector = Speller(lang='en')
        for word in filtered_words:
            correct_word = spell_corrector(word)
            correct_words.append(correct_word)
        # lemmatizing
        wnl = WordNetLemmatizer()
        lemma_words = []
        for word, tag in pos_tag(correct_words):  # Part-of-speech constants for ADJ,VERB,ADV = 'a', 'r', 'v'
            if tag.startswith('JJ'):  # for adjectives
                lemma_word = wnl.lemmatize(word, pos='a')
                lemma_words.append(lemma_word)
            elif tag.startswith('VB'):  # for verbs
                lemma_word = wnl.lemmatize(word, pos='v')
                lemma_words.append(lemma_word)
            elif tag.startswith('RB'):  # for adverbs
                lemma_word = wnl.lemmatize(word, pos='r')
                lemma_words.append(lemma_word)
            else:
                lemma_word = word
                lemma_words.append(lemma_word)

        tweet = " ".join(lemma_words)
        if len(tweet) == 0:  # if after pre-processing sent. has no letter
            tweet = nan_tweet
        # f1.close(),f2.close()
        return tweet

    print('Reading Training Data')
    df_train = pd.read_csv('data/project_training_data_with_class_labels.csv', dtype=str, delimiter=',', quoting=3)
    print(f'train data: {df_train.shape}')
    print(f'{df_train.info()}')
    print(f'check for null value{df_train.isna().sum()}')
    df_train.rename(columns={' Class Labels ': 'Class Labels'}, inplace=True)  # changing col name
    df_train = df_train.replace(to_replace=np.nan, value='', regex=True)
    # Inserting the column before Class Labels col.
    df_train.insert(loc=3,
                    column='Text',
                    value=df_train['Parent Comments'] + " " + df_train['Comments'])

    # df_train.head()
    # we are combining Parent Comment and Comment cols. into one cols.
    df_train['Text'] = df_train['Parent Comments'] + " " + df_train['Comments']

    # df_train.head()
    print('Reading test Data')
    df_test = pd.read_csv('data/project_test_data.csv', dtype=str, delimiter=',', quoting=3)
    print(df_test.head())
    print(df_test.shape)
    # preprocessing of data
    df_test = df_test.replace(to_replace=np.nan, value='', regex=True)
    df_test.rename(columns={'Parent Comments ': 'Parent Comments'}, inplace=True)
    # we are combining Parent Comment and Comment cols. into one cols.
    df_test['Text'] = df_test['Parent Comments'] + " " + df_test['Comments']
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True, nb_workers=64)
    # pre processing on traning data
    print('Preprocess on training data')
    t = time.time()
    xx = df_train['Text'].parallel_apply(preprocess_text)
    df_train.insert(loc=4,
                    column='Pre Processed Text',
                    value=xx)
    # saving
    df_train.to_csv('processed_train.csv', index=False)  # on train data, #*************
    tt = time.time() - t
    print(f'preprocess time {tt} sec')
    ########################
    # on Test data
    print('Preprocess on Test data')
    t = time.time()
    xtt = df_test['Text'].parallel_apply(preprocess_text)  # on test data
    df_test.insert(loc=4,
                   column='Pre Processed Text',
                   value=xtt)
    df_test.to_csv('processed_test.csv', index=False)  # *******************
    tt = time.time() - t
    print(f'preprocess time {tt} sec')
    del df_train, df_test, xx, xtt


####################################
# Embeddings
######################################

def data():
    print('Loading Data')
    # train data
    df_prepro_train = pd.read_csv('processed_train.csv', na_filter=False)
    # test data
    df_prepro_test = pd.read_csv('processed_test.csv', na_filter=False)
    # print(df_prepro_test.shape)
    # df_prepro_test.head()
    # Train data
    X_train = df_prepro_train['Pre Processed Text']
    y_trn = df_prepro_train['Class Labels']

    # Test data
    X_test = df_prepro_test['Pre Processed Text']
    # Without preprocess
    df_wt_pp = df_prepro_train['Text']

    # converting Labels coloumn into numerical form of train data
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_trn)
    return X_train, y, X_test, df_wt_pp


# X_train,y,X_test,df_wt_pp = data()

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean(
                [self.word2vec[w] for w in words if w in self.word2vec.keys()] or
                [np.zeros(self.dim)], axis=0) for words in X
        ])


class MySentences(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for line in self.sentences:
            yield line.split()

    def fit(self):
        return self

    def transform(self):
        return [line.split() for line in self.sentences]


def Classification(X, y, file_name: str, method: str, Gridsearch=False,
                   model: str = 'all', cv=5, w2v=None, dim=None, bow=False, tfidf=False, wv_p=False):  # fs: str = None
    # try:
    #     y_train = y_train.values.ravel()
    #     y_test = y_test.values.ravel()
    # except:
    #     pass
    with open(file_name + '.txt', 'a') as f:
        file_size = os.stat(f'{file_name}.txt').st_size
        if file_size == 0:
            print(f"New File Created: {file_name}.txt")
            f.write("method, Model, Accuracy, f1_micro, f1_macro,other_info \n")
        else:
            print(f"Existing file found: Appending to File: {file_name}.txt")
        f.close()

    # assert fs in ['pca', 'chi2', 'mi', None], f'feature selection should be {{"pca","chi2","mi",None}}. got:{fs}'
    mdls = ['knn', 'lr', 'dtree', 'rf', 'svm', 'mnb', 'all']
    assert model in mdls, f"model should be {{'all','lr','svm','knn','dtree','rf','mnb'}} but got : {model}"
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.20, random_state=3, stratify=y)

    fs = 'chi2'
    classifiers = [
        # KNeighborsClassifier(),
        SGDClassifier(loss='log_loss', n_jobs=-1),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        # AdaBoostClassifier(),
        #         GradientBoostingClassifier(),
        SGDClassifier(n_jobs=-1, early_stopping=True, loss='hinge'),
        MultinomialNB()

    ]
    if Gridsearch == True:
        clf_parameters = [
            #            {
            #                "clf__n_neighbors": [500, 1000],  # np.arange(2,25 ,10),
            #                "clf__metric": ["l1", "cosine"],
            #                "clf__weights": ["uniform", "distance"],
            #            },
            {
                'clf__penalty': ['l1', 'l2'],
                'clf__alpha': [0.0001, 0.001, 0.01]
            },
            {
                'clf__criterion': ["gini", "entropy"],
                # 'clf__max_features':['sqrt', 'log2'],
                'clf__max_depth': [50, 100],
                # 'clf__ccp_alpha':np.logspace(-3,-2,20),#np.logspace(-2.32,-2.3,20),
                "clf__max_leaf_nodes": [100],

                # "clf__splitter" : ["best", "random"],
                # "clf__min_samples_split":np.arange(2,50,10)
            },
            {
                'clf__n_estimators': [80, 100],
                # 'clf__max_features': ['sqrt', 'log2'],
                'clf__max_depth': [50, 100],  # np.arange(4,15,2).tolist(),
                'clf__criterion': ['gini', 'entropy'],
                'clf__bootstrap': [True],
                #         'clf__ccp_alpha':np.logspace(-2,1,10)
            },
            {

                'clf__alpha': [0.001, 0.01, 0.1]
                # 'clf__early_stopping':[True],

            },
            {
                'clf__alpha': [0.001, 0.01, 0.1],

            }
        ]
    else:
        clf_parameters = [{}] * (len(mdls) - 1)
    i = 1

    def fit(i, file_name, X_trn, X_tst, y_trn, y_tst):
        if w2v:
            pipe = Pipeline([
                ('NBCV', MeanEmbeddingVectorizer(w2v, dim)),
                ('nb_norm', MinMaxScaler()),
                ('clf', classifier)
            ])
            feature_parameters = {}
        elif tfidf:
            pipe = Pipeline([
                ('vect', TfidfVectorizer()),
                ('feature_selector', SelectKBest(chi2)),
                ('clf', classifier), ])
            feature_parameters = {
                'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (2, 3)),
                'feature_selector__k': [2000, 10000]  # Unigrams, Bigrams or Trigrams
            }
        elif bow:
            pipe = Pipeline([
                ('vect', CountVectorizer()),
                ('feature_selector', SelectKBest(chi2)),
                ('clf', classifier), ])
            feature_parameters = {
                'vect__min_df': (2, 3),
                # 'vect__max_features':[100000],
                'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (2, 3)),
                'feature_selector__k': [2000, 10000]  # Unigrams, Bigrams or Trigrams
            }
        elif wv_p:
            pipe = Pipeline([
                ('nb_norm', MinMaxScaler()),
                ('clf', classifier)
            ])
            feature_parameters = {}
        else:
            pipe = Pipeline([
                ('clf', classifier)
            ])
            feature_parameters = {}
        parameters = {**feature_parameters, **clf_params}
        # grid = GridSearchCV(pipeline, parameters, scoring='f1_micro', cv=10)
        grid = GridSearchCV(pipe, parameters, scoring='f1_macro', cv=cv, n_jobs=48,
                            verbose=1, pre_dispatch='n_jobs//2')  # early_stopping=False,use_gpu=True)
        try:
            print("_" * 32)
            print(f'{i}.', classifier)
            print("_" * 32)
            if w2v:
                grid.fit(MySentences(X_trn).transform(), y_trn)
                pred = grid.predict(MySentences(X_tst).transform())
            else:
                grid.fit(X_trn, y_trn)
                pred = grid.predict(X_tst)
            best_clf = grid.best_params_
            print(best_clf)
            print(classification_report(y_tst, pred))
            i1 = classifier.__class__.__name__
            i2 = sklearn.metrics.accuracy_score(y_tst, pred)
            i3 = sklearn.metrics.f1_score(y_tst, pred, average='micro')
            i4 = sklearn.metrics.f1_score(y_tst, pred, average='macro')
            with open(f'{file_name}.txt', 'a') as f:
                f.writelines(
                    f'{method},{i1},{i2},{i3},{i4},{f"model:{model};Gridsearch:{Gridsearch};cv:{cv};fs:{fs};best_param:{grid.best_params_}"} \n')
                f.close()
            # dataint['Model'].append(i1)
            # dataint['Accuracy'].append(i2)
            # dataint['f1_micro'].append(i3)
            # dataint['f1_macro'].append(i4)
            print("-" * 80)
            print("-" * 80)
            # data[name] = dataint

        except Exception as e:
            print(e)
        return grid

    if model == 'all':
        for classifier, clf_params in zip(classifiers, clf_parameters):
            fit(i, file_name, X_trn, X_tst, y_trn, y_tst)
            i += 1
    else:
        _ = mdls.index(model)
        # print(_,len(clf_parameters))
        classifier, clf_params = classifiers[_], clf_parameters[_]
        # print(classifier.get_params())
        fit(i, file_name, X_trn, X_tst, y_trn, y_tst)


def Word2Vec(Xtr):
    print('Word2Vec Embeddings')
    sentences = MySentences(Xtr)
    model = Word2Vec(sentences=sentences, vector_size=300, sg=0, window=5, min_count=1, workers=64)
    model.save('word2vec_u300.model')
    del model
    wv = KeyedVectors.load("word2vec_u300.model", mmap='r')
    wv = dict(zip(wv.wv.index_to_key, wv.wv.vectors))
    return wv


def Spcay(Xtr):
    print('Spacy Doc2vec')
    import spacy
    spacy.cli.download("en_core_web_lg")
    import en_core_web_lg
    nlp = en_core_web_lg.load()

    def word2vec_spacy(X):
        w2v_spacy = []
        for item in tqdm(X.values):  # .values returns dataframe rows as list, eg. 1st sentence as ['he is good']
            doc = nlp(item)
            w2v_spacy.append(doc.vector)
        w2v_spacy = np.array(w2v_spacy)
        return w2v_spacy

    # on train data
    w2v_spacy_train = word2vec_spacy(Xtr)
    del nlp
    return w2v_spacy_train


# joblib.dump(w2v_spacy_train, 'w2v_spacy_train.joblib')
# joblib.dump(w2v_spacy_test, 'w2v_spacy_test.joblib')

# w2v_spacy_train,w2v_spacy_test = Spcay(df_wt_pp,X_test)

# Classification(w2v_spacy_train,y,'Final_NLP','Spacy_wp',Gridsearch=True,model='all',bow=False,wv_p=True)

#####################################################

def Glove(Xtr):
    print('Glove Embeddings')
    # load the Stanford GloVe model
    glove_input_file = 'glove/glove.twitter.27B.200d.txt'
    model_glove = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)
    glove_dict = dict(zip(model_glove.index_to_key, model_glove.vectors))
    # on train data

    w2v_glove_train = MeanEmbeddingVectorizer(glove_dict, 200).transform(Xtr)
    del model_glove, glove_dict
    return w2v_glove_train


# w2v_glove_train, w2v_glove_test = Glove(df_wt_pp,X_test)
# Classification(w2v_glove_train,y,'Final_NLP','Glove_wp',Gridsearch=True,model='all',bow=False,wv_p=True)


def best_model(Xtr, ytr, Xte=None, predict=False, ngram_range=(1, 3), min_df=2, k_features=10000, clf_alpha=0.0001,
               clf_penality='l2'):
    print('Training the best model')
    classifier = SGDClassifier(loss='log_loss', n_jobs=-1)
    pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('feature_selector', SelectKBest(chi2)),
        ('clf', classifier), ])
    feature_parameters = {
        'vect__ngram_range': (ngram_range),
        # 'vect__max_features': 100000,
        'vect__min_df': min_df,
        'feature_selector__k': k_features,  # Unigrams, Bigrams or Trigrams
        'clf__penalty': clf_penality,
        'clf__alpha': clf_alpha
    }
    X_trn, X_val, y_trn, y_val = train_test_split(Xtr, ytr, test_size=0.20, random_state=3, stratify=y)
    pipe.set_params(**feature_parameters)
    pipe.fit(X_trn, y_trn)
    y_val_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_val_pred))
    if predict:
        assert Xte != None, 'Pass Xte to predict'
        y_tst_pred = pipe.predict(Xte)
        print("Mapping numeric number to sentiment like Neural, 'non-sarcastic' & 'sarcastic'")
        k = pd.DataFrame(y_tst_pred)
        k[k.columns[0]] = k[k.columns[0]].map({0: 'non-sarcastic', 1: 'sarcastic'})
        # print(k)
        k.to_csv(r'prediction_on_testdata.txt', header=None, index=None, sep='\t')
        return np.array(y_tst_pred)
    return pipe


if __name__ == '__main__':
    # run all models in gridsearch sequentially
    # if preproccess
    # do_preprocess()
    # Loading data from the saved preprocessed file
    X_train, y, X_test, df_wt_pp = data()

    ################################
    # best model
    best_model(X_train, y)
    ################################

    # ---------------------------------------------------
    ############################################
    # ******* with preprocessing *********
    ############################################
    print('With preprocess')

##    # 1. Bag of words
#    print('BoW with preprocess')
##    Classification(X_train,y,'Final_NLP','BOW',Gridsearch=True,model='all',bow=True)
##
##    # 2. Tfidf
#    print('Tfidf with preprocess')
#    Classification(X_train,y,'Final_NLP','Tfidf',Gridsearch=True,model='all',tfidf=True)
##
##    # 3. Word2Vec trained
#    print('Word2vec with preprocess')
##    wv = Word2Vec(X_train)
##    Classification(X_train,y,'Final_NLP','W2V',Gridsearch=True,model='all',w2v=wv,dim=300)
##    del wv
##
##    # # 4. Word2Vec Spacy pretrained
#    print('Spacy with preprocess')
##    w2v_spacy_train = Spcay(X_train)
##    Classification(w2v_spacy_train,y,'Final_NLP','Spacy',Gridsearch=True,model='all',bow=False,wv_p=True)
##    del w2v_spacy_train
##
##    # # 5. Glove Pretrained Embeddings
#    print('Glove with preprocess')
##    w2v_glove_train = Glove(X_train)
##    Classification(w2v_glove_train,y,'Final_NLP','Glove',Gridsearch=True,model='all',bow=False,wv_p=True)
##    del w2v_glove_train
##
##    #############################
##    # Without Preprocess
##    #############################
##    print('Without preprocess')
##
##    # 1. Bag of words
#    print('BoW without preprocess')
##    Classification(df_wt_pp,y,'Final_NLP','BOW_wp',Gridsearch=True,model='all',bow=True)
##
##    # 2. Tfidf
#    print('Tfidf without preprocess')
##    Classification(df_wt_pp,y,'Final_NLP','Tfidf_wp',Gridsearch=True,model='all',tfidf=True)
##
##    # 3. Word2Vec trained
#    print('Word2vec without preprocess')
##    wv = Word2Vec(df_wt_pp)
##    Classification(df_wt_pp,y,'Final_NLP','W2V_wp',Gridsearch=True,model='all',w2v=wv,dim=300)
##    del wv
##    # 4. Word2Vec Spacy pretrained
#    print('Spacy without preprocess')
##    w2v_spacy_train = Spcay(df_wt_pp)
##    Classification(w2v_spacy_train,y,'Final_NLP','Spacy_wp',Gridsearch=True,model='all',bow=False,wv_p=True)
##    del w2v_spacy_train
#
##    # 5. Glove Pretrained Embeddings
#    print('Glove without preprocess')
##    w2v_glove_train = Glove(df_wt_pp)
##    Classification(w2v_glove_train,y,'Final_NLP','Glove_wp',Gridsearch=True,model='all',bow=False,wv_p=True)
##    del w2v_glove_train
##
##    print('END')