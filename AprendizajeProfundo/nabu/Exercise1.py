
# coding: utf-8

# In[1]:


# Exercise 1
import pandas
import keras
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Activation, Dense, Dropout

from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding

from keras.models import Sequential
from keras import optimizers, regularizers
import matplotlib.pyplot as plt

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


# In[2]:


def load_dataset():
    dataset = load_files('dataset/txt_sentoken/', shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)
    print('\nTraining samples {}, test_samples {}:'.format(len(X_train), len(X_test)))


    
    vectorizer = TfidfVectorizer(binary=True, ngram_range=(1, 2), stop_words='english', max_df=0.7, norm='l2', vocabulary=None)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test


# In[3]:


def plot_history(history):
  
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plot_loss = plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig("plot_loss.png")
    
    plt.close(plot_loss)
    
    ## Accuracy
    plot_accuracy = plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
   
    plt.savefig("plot_accuracy.png")
    plt.close(plot_accuracy)

    # comment when run on queue
    # plt.show()


# In[4]:


X_train, X_test, y_train, y_test = load_dataset()


train_examples = X_train.shape[0]
input_size = X_train.shape[1]
test_examples = X_test.shape[0]


batch_size = 50
num_classes = 1
epochs = 5

print(train_examples)
print(test_examples)
print(input_size)


# In[5]:


X_train = X_train.reshape(train_examples, input_size)
X_test = X_test.reshape(test_examples, input_size)

# model = Sequential()
# model.add(Dense(512, input_shape=(input_size,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))
# model.add(Activation('softmax'))

model = Sequential()
model.add(Dense(512, input_shape=(input_size,)))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
      optimizer='adam', 
      metrics=['accuracy']) 


# In[6]:


plot_model(model, to_file='model.png')

# comment when run on queue
# SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[7]:


hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1, );


# In[8]:


scores = model.evaluate(X_test, y_test)


# In[9]:


plot_history(hist) 


# In[10]:


print('\n')
print('Loss:', scores[0])
print('Accuracy:', scores[1])


# In[11]:


predictions = model.predict(X_test)
prediction_int = [round(x[0]) for x in predictions]


# In[12]:


results = pandas.DataFrame(y_test, columns=['true_value'])
results.loc[:, 'prediction'] = predictions
results.loc[:, 'prediction_int'] = prediction_int
results.to_csv("predictions.csv",index=False)


# In[13]:


model.save_weights("model_weights.h5")


# In[14]:


model.save("model.h5")

