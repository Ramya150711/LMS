#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv('emotion.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.label.value_counts()


# In[6]:


import seaborn as sns 
sns.countplot(x=df.label)


# In[8]:


df.isna().sum()


# In[11]:


df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[12]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
df['text']=df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[13]:


get_ipython().system('pip install textblob')


# In[18]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
df['text']=df['text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
df['text'].head()


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(df['text'])
x=x.toarray()
y=df.label.values


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)


# In[24]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model=model.fit(x_train,y_train)
pred=model.predict(x_test)


# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[26]:


print(confusion_matrix(y_test,pred))


# In[27]:


print(accuracy_score(y_test,pred))


# In[29]:


print(classification_report(y_test,pred))


# In[32]:


from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()
clf_rf.fit(x_train,y_train)
rf_pred=clf_rf.predict(x_test).astype(int)


# In[33]:


print(confusion_matrix(y_test,rf_pred))


# In[34]:


print(accuracy_score(y_test,rf_pred))


# In[35]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
lr_pred=logreg.predict(x_test)


# In[36]:


print(confusion_matrix(y_test,lr_pred))


# In[37]:


print(classification_report(y_test,lr_pred))


# In[38]:


print(accuracy_score(y_test,lr_pred))


# # conclusion

# Random Forest model proved better with the accuracy 88% when compared other models
