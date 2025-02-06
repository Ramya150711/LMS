#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
dataset=pd.read_csv('hate_speech.csv')
dataset.head()


# In[22]:


for index,tweet in enumerate(dataset['tweet'][10:15]):
    print(index+1,"-",tweet)


# In[23]:


import re
def clean_text(text):
    text=re.sub(r'[^a-zA-Z\']',' ',text)
    text=re.sub(r'[^\x00-\x7F]+',' ',text)
    text=text.lower()
    return text


# In[24]:


dataset['clean_text']=dataset.tweet.apply(lambda x:clean_text(x))


# In[25]:


dataset.head(10)


# In[28]:


from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(strip_accents=None,
                      lowercase=False,
                      preprocessor=None,
                      tokenizer=tokenizer_porter,
                      use_idf=True,
                      norm='l2',
                      smooth_idf=True)
x=tfidf.fit_transform(dataset['clean_text'])
x=x.toarray()
y=dataset.label.values


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=27,\
                                              test_size=0.2,shuffle=True)


# In[33]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model=model.fit(x_train,y_train)
pred=model.predict(x_test)


# In[ ]:




