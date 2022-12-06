import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer, Embedding, Input, GlobalAveragePooling1D, Dense, Flatten, SimpleRNN, GlobalMaxPooling1D,  LSTM, SpatialDropout1D, Activation

from keras.models import Sequential, Model
from keras.callbacks import  EarlyStopping
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# # train data
# df_prepro_train = pd.read_csv('/content/drive/MyDrive/processed_train.csv',na_filter=False)
# # Train data
# X_trn = df_prepro_train['Pre Processed Text']
# y_trn = df_prepro_train['Class Labels']


print('Loading Data')
# train data
df_prepro_train = pd.read_csv('processed_train.csv',na_filter=False)
# test data
df_prepro_test = pd.read_csv('processed_train.csv',na_filter=False)
# print(df_prepro_test.shape)
# df_prepro_test.head()
# Train data
X_trn = df_prepro_train['Pre Processed Text']
y_trn = df_prepro_train['Class Labels']

# Test data
X_test = df_prepro_test['Pre Processed Text']


vocab_size = 5000  # Only consider the top 5k words
maxlen = 200  # Only consider the first 200 words of each sequence
embed_dim = 200# Embedding size for each token 200
# pre-processing step of vectorization so they can be used in NN.
def preprocess_data(train_dataset,test_dataset=None):
    print('Doing preprocess')
    """
    * Build tokenizer using the training set.

    * Convert the word sequences in each sentence into integer with the tokenizer.

    * Pad input lengths to uniform sizes : tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen)
        this function transforms a list of num_samples sequences (lists of integers) into a 2D Numpy array 
        of shape (num_samples, num_timesteps). num_timesteps is the maxlen argument.
         sequences should be in form [[1,56,78],[34,7,89]]

        Sequences that are shorter than num_timesteps are padded with value.
        Sequences longer than num_timesteps are truncated so that they fit the desired length.
        
    """
    ## on training data
    
    # tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    # fit tokenizer
    tokenizer.fit_on_texts(train_dataset)
    # text to sequences
    train_texts_to_int = tokenizer.texts_to_sequences(train_dataset)   # list type; each element in this list have seq. nos. of all words 
    # of each sentence
    # Pad sequences to the same length.
    train_int_texts_to_pad = tf.keras.preprocessing.sequence.pad_sequences(train_texts_to_int, maxlen=maxlen)

    test_texts_to_int = tokenizer.texts_to_sequences(test_dataset)
    test_int_texts_to_pad = tf.keras.preprocessing.sequence.pad_sequences(test_texts_to_int, maxlen=maxlen)
    x_test = test_int_texts_to_pad
    
    # x_train, x_valid, x_test
    x_train = train_int_texts_to_pad
    # x_valid = valid_int_texts_to_pad
 
    
    
    # Check Total Vocab Size
    total_vocab_size = len(tokenizer.word_index) + 1
    print('Total Vocabulary Size (Untrimmed): %d' % total_vocab_size)
    print('Vocabulary Size (trimmed): %d' % vocab_size)
    
    return x_train, x_test


x_train,  X_test = preprocess_data(X_trn,X_test)

print('x_train.shape', x_train.shape)

# encoder = LabelEncoder()
# Y = encoder.fit_transform(y_trn)
#Converting categorical labels to numbers.
Y = pd.get_dummies(y_trn).values
print('Shape of label tensor:', Y.shape)

# print('no. of classes : ', encoder.classes_)

#Splitting Train / Valid
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_valid.shape,Y_valid.shape)


# Transformer block
class TransformerBlock(Layer):
    # initialization of various layers
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) # multi-head attention layer
        # feed-forward network
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # batch normalization
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)   # dropout layers
        self.dropout2 = Dropout(rate)
    
    # defined the layers according to the architecture of the transformer block
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # multi-head attention layer
        attn_output = self.dropout1(attn_output, training=training)  # 1st dropout
        out1 = self.layernorm1(inputs + attn_output)  # 1st normalization
        ffn_output = self.ffn(out1)   # feed-forward network
        ffn_output = self.dropout2(ffn_output, training=training)  # 2nd dropout
        out2 = self.layernorm2(out1 + ffn_output)  # 2nd normalization
        return out2


# positional embeddings
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim) # embedding layer for tokens
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)  # embedding layer for token positions

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



num_heads = 12  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block1(x)
# x = transformer_block2(x)
# x = transformer_block3(x)
x = GlobalAveragePooling1D()(x)


x = Dropout(0.1)(x)
x = Dense(128, activation="relu")(x)
# x = Dense(128, activation="relu")(x)
# x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary() # print model summary

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy",'Precision','Recall'])
history = model.fit(X_train, Y_train, 
                    batch_size=4096//32, epochs=100, 
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3,verbose = 1)]
                   )

result_trsfmr = model.evaluate(X_valid,Y_valid,batch_size = 512)
print(f'Valid set Transformer MODEL\n  Loss: {result_trsfmr[0]:0.3f}\n  Accuracy: {result_trsfmr[1]:0.3f}')


predct = np.argmax(model.predict(X_test,batch_size = 512),axis = 1)
ytest_predicted =predct.flatten()
print(ytest_predicted[0:10])
k = pd.DataFrame(ytest_predicted)
k[k.columns[0]] = k[k.columns[0]].map({0: 'non-sarcastic', 1: 'sarcastic'})
# print(k)
k.to_csv(r'prediction_transformer.txt', header=None, index=None, sep='\t')
print('----- END ----')

del model




###########  FFNN ###################
ffnn = Sequential() 
ffnn.add(Embedding(vocab_size,embed_dim, input_length=maxlen))
ffnn.add(Flatten()) 
ffnn.add(Dense(64, activation='relu')) 
ffnn.add(Dense(64, activation='relu')) 
ffnn.add(Dense(64, activation='relu'))
ffnn.add(Dense(2, activation='softmax'))

## Compile the keras model
ffnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Precision', 'Recall']
              )
ffnn.summary()

## Fit keras model on the dataset
ffnn.fit(X_train, Y_train, 
          batch_size=4096//32,
          epochs=100
          ,callbacks=[EarlyStopping(monitor='val_loss', patience=5,verbose = 1)],validation_split = 0.1)


result_ffnlm = ffnn.evaluate(X_valid,Y_valid,batch_size = 512)
print(f'Valid set FFNN MODEL\n  Loss: {result_ffnlm[0]:0.3f}\n  Accuracy: {result_ffnlm[1]:0.3f}')

predct = np.argmax(ffnn.predict(X_test,batch_size = 512),axis=1)
ytest_predicted = predct.flatten()
k = pd.DataFrame(ytest_predicted)
k[k.columns[0]] = k[k.columns[0]].map({0: 'non-sarcastic', 1: 'sarcastic'})
# print(k)
k.to_csv(r'prediction_ffnn.txt', header=None, index=None, sep='\t')
print('----- END ----')

del ffnn



#######   LSTM  ##########
Stacklstm = Sequential()
Stacklstm.add(Embedding(vocab_size,embed_dim, input_length=maxlen))
Stacklstm.add(LSTM(128, return_sequences=True))
Stacklstm.add(LSTM(128, return_sequences=True)) 
Stacklstm.add(LSTM(64, return_sequences=True)) 
Stacklstm.add(LSTM(32)) 
Stacklstm.add(Dense(2, activation='softmax'))
Stacklstm.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Precision', 'Recall']
              )
Stacklstm.summary()


## Fit keras model on the dataset
Stacklstm.fit(X_train, Y_train, 
          batch_size=4096//32,
          epochs=100
          ,callbacks=[EarlyStopping(monitor='val_loss', patience=5,verbose = 1)],validation_split = 0.1)


result_lstm = Stacklstm.evaluate(X_valid,Y_valid,batch_size = 1024)
print(f'Valid set LSTM MODEL\n  Loss: {result_lstm[0]:0.3f}\n  Accuracy: {result_lstm[1]:0.3f}')

predct = np.argmax(Stacklstm.predict(X_test,batch_size = 1024),axis=1)
ytest_predicted = predct.flatten()
k = pd.DataFrame(ytest_predicted)
k[k.columns[0]] = k[k.columns[0]].map({0: 'non-sarcastic', 1: 'sarcastic'})
# print(k)
k.to_csv(r'prediction_lstm.txt', header=None, index=None, sep='\t')
print('----- END ----')

del Stacklstm




#######  RNN  ########
stackrnn = Sequential()
stackrnn.add(Embedding(vocab_size,embed_dim, input_length=maxlen))
stackrnn.add(SimpleRNN(100, return_sequences=True,unroll=True)) 
stackrnn.add(SimpleRNN(100, return_sequences=True,unroll=True))
stackrnn.add(SimpleRNN(64, return_sequences=True,unroll=True)) 
stackrnn.add(SimpleRNN(64,unroll=True)) 
stackrnn.add(Dense(2, activation='softmax'))
stackrnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Precision', 'Recall']
              )
stackrnn.summary()

## Fit keras model on the dataset
stackrnn.fit(X_train, Y_train, 
          batch_size=4096//16,
          epochs=100
          ,callbacks=[EarlyStopping(monitor='val_loss', patience=5,verbose = 1)],validation_split = 0.1)

result_rnn = stackrnn.evaluate(X_valid,Y_valid,batch_size = 512)
print(f'Valid set RNN MODEL\n  Loss: {result_rnn[0]:0.3f}\n  Accuracy: {result_rnn[1]:0.3f}')

predct = np.argmax(stackrnn.predict(X_test,batch_size = 512),axis=1)
ytest_predicted = predct.flatten()
k = pd.DataFrame(ytest_predicted)
k[k.columns[0]] = k[k.columns[0]].map({0: 'non-sarcastic', 1: 'sarcastic'})
# print(k)
k.to_csv(r'prediction_rnn.txt', header=None, index=None, sep='\t')
print('----- END ----')


del stackrnn

################################### 



if __name__ == '__main__':
    pass