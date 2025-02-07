#!/usr/bin/env python
# coding: utf-8

# In[1]:


text="I am learning NLP"


# In[3]:


import pandas as pd
pd.get_dummies(text.split())


# In[5]:


text=["I love NLP and i will learn NLP in 2 months"]


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
vectorizer.fit(text)
vector=vectorizer.transform(text)


# In[8]:


print(vectorizer.vocabulary_)
print(vector.toarray())


# In[9]:


print(vector)


# In[10]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[11]:


df=pd.DataFrame(data=vector.toarray(),columns=vectorizer.get_feature_names_out())
df


# In[12]:


text='I am learning NLP'


# In[13]:


from textblob import TextBlob
TextBlob(text).ngrams(1)


# In[14]:


TextBlob(text).ngrams(2)


# In[17]:


TextBlob(text).ngrams(3)


# In[18]:


TextBlob(text).ngrams(4)


# In[20]:


text=['The quick brown box for jumped over the lazy dog','The dog.','The fox']


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)
print(vectorizer.idf_)


# In[ ]:




