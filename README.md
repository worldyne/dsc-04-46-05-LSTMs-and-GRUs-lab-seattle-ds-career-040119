
# LSTMs and GRUs - Lab

## Introduction

In this lab, we'll learn how to use LSTM cells and GRU cells to build **_Recurrent Neural Networks_** to work with text data!

## Objectives

You will be able to:

* Explain the the problem of vanishing and exploding gradients, and why they are a problem when training RNNs
* Demonstrate an understanding of the basic architecture and function of a Long Short Term Memory cell
* Demonstrate an understanding of the basic architecture and function of a Gated Recurrent Unit

### Getting Started

In this lab, we'll see a basic example of how we can use LSTMs and GRU cells to build a Recurrent Neural Network for text classification on the _Newsgroups Dataset_ that is included with scikit-learn. The goal of this lab build 2 nearly identical models so that we can benchmark performances for both LSTMs and GRUs and compare them against one another. 

We'll begin by loading in everything we'll need for this lab. 

In the cell below, import the following items:

* `fetch20_newsgroups`, from `sklearn.datasets`
* `keras`
* from `keras.layers`, import the following layers:
    * `LSTM`
    * `GRU`
    * `Dense`
    * `GlobalMaxPool1D`
    * `Embedding`
    * `Dropout`
* `Sequential`, from `keras.models`
* `text` and `sequence`, from `keras.preprocessing`
* `numpy`, `matplotlib`, and `pandas`. Set the standard alias for each.
* Also set matplotlib visualizations to appear inline, and use numpy to set a random seed of `0`.


```python
from sklearn.datasets import fetch_20newsgroups
import keras
from keras.layers import LSTM, GRU, Dense, GlobalMaxPool1D, Embedding, Dropout
from keras.preprocessing import text, sequence
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(0)
import pandas as pd
```

    Using TensorFlow backend.
    

### Importing and Preprocessing Our Text Data

Since we'll be working with a text dataset, we'll need to do a few things to get it into a format where our LSTM and GRU networks can work with it. Specifically, we'll need to:

* Import and load the data and labels, and store them separately
* Convert the labels to a one-hot encoded format
* tokenize our text data
* Convert the tokenized text to sequences
* Pad the sequences, so that they are all the same length. 

Let's start by loading in our data. In the cell below, call `fetch_20newsgroups()` to get our data and labels.


```python
newsgroups = fetch_20newsgroups()
```

Now, let's split off our data and labels, which are currently stored in our `newgroups` object's `.data` and `.target` attributes, respectively.  

In the cell below, store the `data` and the `target` in the appropriate variables.


```python
data = newsgroups.data
labels = newsgroups.target
```

Next, we'll need to convert our data to a one-hot encoded format. Keras has a utility function that can easily do this for us called `to_categorical()`, which can be found in `keras.utils`.

In the cell below, call the `to_categorical()` function and pass in `labels`, as well as the number of unique classes in our labels, which is `20`.


```python
labels = keras.utils.to_categorical(labels, 20)
```

#### Creating Sequences From Text

By now, you've seen this code before. Anytime we work with text data for deep learning, you can expect to see the following preprocessing pattern:

> **raw text --> tokenized text --> text sequences --> padded sequences**

In the cell below:

* Instantiate a `Tokenizer` object, which can be found in the `text` module that we've already imported from `keras`. Set the `num_words` parameter to `20000`, so that our model only keeps the 20,000 must used words.
* Call the `tokenizer` object's `.fit_on_texts()` method, and pass in our `data`, which should be converted to a python `list` (do this right inside the method call)
* Next, call the `tokenizer` object's `texts_to_sequences()` method and pass in our `data`.
* Finally, use the `sequence` module's `pad_sequences()` method to make sure all of our sequences are padded to the exact same size, so that we can set hard limits on the dimensionality of our inputs. For input, pass in our `list_tokenized_train`, as well as the parameter `maxlen=100`, so that all sequences are padded to be of length 100, regardless of the amount of text they actually contain. 


```python
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(data))
list_tokenized_train = tokenizer.texts_to_sequences(data)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=100)
```

Great! We've now finished preprocessing our data, and we're ready to build, compile, and train our models!

### Creating Our Models

Now, to the fun part--creating, and training to similar LSTM and GRU networks, and comparing their performance and runtimes. 

#### Architectures

Both of our models will stick to the following architecture:

1. An `Embedding()` layer, of size `(20000, 128)`. This means that the first parameter passed into the embedding layer should be `20000` for the 20,000 words in our our text vocabulary, and the second parameter should be `128`, for the size of the Dense vectors the embedding layer will learn for each of the 20,000 words. 
2. An `LSTM()` layer (or `GRU()` layer, for the second model) of size `50`. During this step, also set the `return_sequences` parameter to `True`, so that during back propagation our models will calculate loss and learn for every step of the sequence, not just the final result of the sequence.
3. A `GlobalMaxPool1D()` layer, so that our model performs a combined _MaxPool_  operation across all weights in the recurrent layer. 
4. A `Dropout()` layer set to `0.5`.
5. A `Dense()` layer of size `50`, with this layer's `activation` parameter set to `'relu'`
6. Another `Dropout()` layer set to `0.5`
7. A `Dense()` layer that will act as our output layer. This layer should contain `20` neurons (one for each possible predicted class), and should have it's `activation` parameter set to `'softmax'`

In the cell below, create our `LSTM` model. 

**_NOTE:_** For simplicity's sake, we recommend you make use a `Sequential()` object and use that object's `.add()` parameter to add each layer to the network. 


```python
# LSTM Model

lstm_model = Sequential()
lstm_model.add(Embedding(20000, 128))
lstm_model.add(LSTM(50, return_sequences=True))
lstm_model.add(GlobalMaxPool1D())
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(50, activation='relu'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20, activation='softmax'))
```

#### Compilation Parameters

Now that we've built our model, we still need to compile it. 

In the cell below, call our model's `.compile()` method and pass in the following parameters:

* `loss='categorical_crossentropy'`
* `optimizer='adam'`
* `metrics=['accuracy']`


```python
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

#### Inspecting Our Compiled Model

Before we train our model, let's take a look at what it looks like, and see how many trainable parameters it has. In the cell below, call our model's `.summary()` method to inspect it. 


```python
lstm_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, None, 128)         2560000   
    _________________________________________________________________
    lstm_1 (LSTM)                (None, None, 50)          35800     
    _________________________________________________________________
    global_max_pooling1d_1 (Glob (None, 50)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 50)                2550      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 20)                1020      
    =================================================================
    Total params: 2,599,370
    Trainable params: 2,599,370
    Non-trainable params: 0
    _________________________________________________________________
    

Just under 2.6 million trainable parameters--that's a pretty decent-sized model!

#### Training Our Model

Now that we have preprocessed our data, created our model, and compiled it, we're ready for the moment of truth--training!

In the cell below, call our model's `.train()` method and pass in the following parameters:

* `X_t`, our padded sequence data
* `labels`
* `epochs=2`
* `batch_size=32`, so that our model trains on mini-batches of 32 examples at a time.
* `validation_data=0.1`, so that our model hold out 10% of our data for validation.

**_NOTE:_** This will take a few minutes per epoch to train!


```python
lstm_model.fit(X_t, labels, epochs=2, batch_size=32, validation_split=0.1)
```

    Train on 10182 samples, validate on 1132 samples
    Epoch 1/2
    10182/10182 [==============================] - 52s 5ms/step - loss: 2.9184 - acc: 0.0855 - val_loss: 2.5967 - val_acc: 0.1749
    Epoch 2/2
    10182/10182 [==============================] - 50s 5ms/step - loss: 2.2823 - acc: 0.2511 - val_loss: 1.7920 - val_acc: 0.5141
    




    <keras.callbacks.History at 0x23438ce46a0>



### Building Our GRU Model

Now that we have a benchmark for how an LSTM model performs, let's build the exact same model, but with `GRU()` cells instead of `LSTM()` cells!

In the cell below, recreate the network we did above, but with `GRU()` neurons immediately following our Embedding layer instead of `LSTM` cells. Use the exact same parameters as we did above at each layer--we want things to be as equal as possible between them, so that we get a good baseline for comparing the two models on performance and runtime!


```python
# GRU Model

gru_model = Sequential()
gru_model.add(Embedding(20000, 128))
gru_model.add(GRU(50, return_sequences=True))
gru_model.add(GlobalMaxPool1D())
gru_model.add(Dropout(0.5))
gru_model.add(Dense(50, activation='relu'))
gru_model.add(Dropout(0.5))
gru_model.add(Dense(20, activation='softmax'))
```

Now, compile the model with the same parameters we did for the first network.


```python
gru_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Now, let's look at a `.summary()` of our GRU model, and see if it has more or less total trainable parameters than our LSTM model. 


```python
gru_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, None, 128)         2560000   
    _________________________________________________________________
    gru_1 (GRU)                  (None, None, 50)          26850     
    _________________________________________________________________
    global_max_pooling1d_2 (Glob (None, 50)                0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 50)                2550      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 20)                1020      
    =================================================================
    Total params: 2,590,420
    Trainable params: 2,590,420
    Non-trainable params: 0
    _________________________________________________________________
    


```python
gru_model.fit(X_t, labels, epochs=2, batch_size=32, validation_split=0.1)
```

    Train on 10182 samples, validate on 1132 samples
    Epoch 1/2
    10182/10182 [==============================] - 41s 4ms/step - loss: 2.9062 - acc: 0.1079 - val_loss: 2.5377 - val_acc: 0.3207
    Epoch 2/2
    10182/10182 [==============================] - 40s 4ms/step - loss: 2.2111 - acc: 0.3149 - val_loss: 1.7253 - val_acc: 0.5256
    




    <keras.callbacks.History at 0x23438c236a0>



There we have it! In this particular case, GRUs strongly outperformed LSTMs in the first epoch, but the gap quickly leveled out between them by the end of epoch 2. When comparing LSTMs and GRUs for a given task, this isn't always the case--there are certainly times where LSTMs will outperform GRUs. However, overall, GRUs seem to have a slight advantage over LSTMs. The interesting thing about this is that researchers don't yet know _why_ GRUs tend to slightly outperform LSTMs, especially when GRU cells are a bit simpler than LSTM cells. This is an ongoing area of cutting-edge research in the field of Deep Learning--maybe someday, you'll be the one to solve this mystery!

# Summary

In this lesson, we created and trained comparable LSTM and GRU models for text classification, and compared their performance and runtimes against one another!
