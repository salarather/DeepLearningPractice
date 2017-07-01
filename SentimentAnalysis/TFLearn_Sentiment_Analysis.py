
# coding: utf-8

# # Sentiment analysis with TFLearn
# 


# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical


# ## Preparing the data
# 
# Our goal here is to convert our reviews into word vectors. The word vectors will have elements representing words in the total vocabulary. If the second position represents the word 'the', for each review we'll count up the number of times 'the' appears in the text and set the second position to that count. I'll show you examples as we build the input data from the reviews data. 

# ### Read the data
# 
# Use the pandas library to read the reviews and postive/negative labels from comma-separated files. The data we're using has already been preprocessed a bit and we know it uses only lower case characters. If we were working from raw data, where we didn't know it was all lower case, we would want to add a step here to convert it. That's so we treat different variations of the same word, like `The`, `the`, and `THE`, all the same way.

# In[2]:


reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)


# ### Counting word frequency
# 
# 
# In[3]:


from collections import Counter
total_counts = Counter()
for _, row in reviews.iterrows():
    total_counts.update(row[0].split(' '))
print("Total words in data set: ", len(total_counts))


# Let's keep the first 10000 most frequent words. Most of the words in the vocabulary are rarely used so they will have little effect on our 
# predictions. Below, we'll sort `vocab` by the count value and keep the 10000 most frequent words.

# In[4]:


vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])



# In[5]:


print(vocab[-1], ': ', total_counts[vocab[-1]])





word2idx = {word: i for i, word in enumerate(vocab)} ## create the word-to-index dictionary here


# ### Text to vector function
# 
# Now we can write a function that converts a some text to a word vector. The function will take a string of words as input and 
# return a vector with the words counted up.

# In[7]:


def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)


#    

# In[8]:


text_to_vector('The tea is for a party to celebrate '
               'the movie so she has no time for a cake')[:65]


# Now, run through our entire review data set and convert each review to a word vector.

# In[9]:


word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])


# In[10]:


# Printing out the first 5 word vectors
word_vectors[:5, :23]


# split the data into validation test and trraining sets
# In[11]:


Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)


# In[12]:


trainY



# In[22]:


# Network building
def build_model():
     # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    # Inputs
    net = tflearn.input_data([None, 10000])

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')

    # Output layer
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', 
                             learning_rate=0.01, 
                             loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model


# ## Intializing the model
# 
# Next we need to call the `build_model()` function to actually build the model. In my solution I haven't included any arguments to the function, but you can add arguments so you can change parameters in the model if you want.
# 
# > **Note:** You might get a bunch of warnings here. TFLearn uses a lot of deprecated code in TensorFlow. Hopefully it gets updated to the new TensorFlow version soon.

# In[23]:


model = build_model()


# ## Training the network

# In[24]:


# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=200)


# ## Testing
# 
# 
# In[ ]:


predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)


# ## Try out your own text!

# In[ ]:


# Helper function that uses your model to predict sentiment
def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob), 
          'Positive' if positive_prob > 0.5 else 'Negative')


# In[ ]:


sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence)

sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence)

