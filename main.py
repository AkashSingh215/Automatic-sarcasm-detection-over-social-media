from nlpteest import *
if __name__ == '__main__':
# run all models in gridsearch sequentially
    # if preproccess
    # do_preprocess()
    # Loading data from the saved preprocessed file
    X_train,y,X_test,df_wt_pp = data()

    ################################
    # best model
    #best_model(X_train,y)
    ################################

    #---------------------------------------------------
    ############################################
    # ******* with preprocessing *********
    ############################################
    print('With preprocess')

#    # 1. Bag of words
    print('BoW with preprocess')
#    Classification(X_train,y,'Final_NLP','BOW',Gridsearch=True,model='all',bow=True)
#
#    # 2. Tfidf
    print('Tfidf with preprocess')
    Classification(X_train,y,'Final_NLP','Tfidf',Gridsearch=True,model='all',tfidf=True)
#
#    # 3. Word2Vec trained
    print('Word2vec with preprocess')
#    wv = Word2Vec(X_train)
#    Classification(X_train,y,'Final_NLP','W2V',Gridsearch=True,model='all',w2v=wv,dim=300)
#    del wv
#    
#    # # 4. Word2Vec Spacy pretrained
    print('Spacy with preprocess')
#    w2v_spacy_train = Spcay(X_train)
#    Classification(w2v_spacy_train,y,'Final_NLP','Spacy',Gridsearch=True,model='all',bow=False,wv_p=True)
#    del w2v_spacy_train
#    
#    # # 5. Glove Pretrained Embeddings
    print('Glove with preprocess')
#    w2v_glove_train = Glove(X_train)
#    Classification(w2v_glove_train,y,'Final_NLP','Glove',Gridsearch=True,model='all',bow=False,wv_p=True)
#    del w2v_glove_train
#
#    #############################
#    # Without Preprocess
#    #############################
#    print('Without preprocess')
#
#    # 1. Bag of words
    print('BoW without preprocess')
#    Classification(df_wt_pp,y,'Final_NLP','BOW_wp',Gridsearch=True,model='all',bow=True)
#
#    # 2. Tfidf
    print('Tfidf without preprocess')
#    Classification(df_wt_pp,y,'Final_NLP','Tfidf_wp',Gridsearch=True,model='all',tfidf=True)
#
#    # 3. Word2Vec trained
    print('Word2vec without preprocess')
#    wv = Word2Vec(df_wt_pp)
#    Classification(df_wt_pp,y,'Final_NLP','W2V_wp',Gridsearch=True,model='all',w2v=wv,dim=300)
#    del wv
#    # 4. Word2Vec Spacy pretrained
    print('Spacy without preprocess')
#    w2v_spacy_train = Spcay(df_wt_pp)
#    Classification(w2v_spacy_train,y,'Final_NLP','Spacy_wp',Gridsearch=True,model='all',bow=False,wv_p=True)
#    del w2v_spacy_train

#    # 5. Glove Pretrained Embeddings
    print('Glove without preprocess')
#    w2v_glove_train = Glove(df_wt_pp)
#    Classification(w2v_glove_train,y,'Final_NLP','Glove_wp',Gridsearch=True,model='all',bow=False,wv_p=True)
#    del w2v_glove_train
#    
#    print('END')