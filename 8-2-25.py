#!/usr/bin/env python
# coding: utf-8

# In[19]:


from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[20]:


df=pd.read_csv('covid_fake.csv')


# In[21]:


df.head()


# In[22]:


df.shape


# In[23]:


df['label'].value_counts()


# In[24]:


df.loc[5:15]


# In[25]:


df.isna().sum()


# In[26]:


df.loc[df['label']=='FAKE',['label']]='FAKE'
df.loc[df['label']=='fake',['label']]='FAKE'
df.loc[df['source']=='facebook',['label']]='Facebook'
df.text.fillna(df.title,inplace=True)
df.loc[5]['label']='FAKE'
df.loc[15]['label']='TRUE'
df.loc[43]['label']='FAKE'
df.loc[131]['label']='TRUE'
df.loc[242]['label']='FAKE'
df.title.fillna('missing',inplace=True)
df.source.fillna('missing',inplace=True)
df['title_text']=df['title']+' '+df['text']


# In[27]:


df.isna().sum()


# In[28]:


df['label'].value_counts()


# In[29]:


df.head()


# In[30]:


df.shape


# In[31]:


df['title_text'][3]


# In[33]:


def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    text=re.sub(r'[^\w\s]','',text)
    text=re.sub(r'[\n]','',text)
    text=text.lower()
    return text

df['title_text']=df['title_text'].apply(preprocessor)
df['title_text'][3]


# In[15]:


porter=PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[36]:


tfidf=TfidfVectorizer(strip_accents=None,
                      lowercase=False,
                      preprocessor=None,
                      tokenizer=tokenizer_porter,
                      use_idf=True,
                      norm='l2',
                      smooth_idf=True)
x=tfidf.fit_transform(df['title_text'])
y=df.label.values


# In[17]:


x.shape


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,\
                                              test_size=0.3,shuffle=False)


# In[38]:


clf=LogisticRegressionCV(cv=5,scoring='accuracy',random_state=0,n_jobs=-1,\
                        verbose=0,max_iter=300)
clf.fit(x_train,y_train)
fake_news_model=open('fake_news_model.sav','wb')
pickle.dump(clf,fake_news_model)
fake_news_model.close()


# In[42]:


filename='fake_news_model.sav'
saved_clf=pickle.load(open(filename,'rb'))
saved_clf.score(x_test,y_test)


# In[41]:


from sklearn.metrics import classification_report,accuracy_score
y_pred=clf.predict(x_test)
print("---Test Set Results---")
print(classification_report(y_test,y_pred))


# In[43]:


clf.predict(x_test[59])


# In[44]:


test="Corona virus before it reaches the lungs"
inp=[test]
vect=tfidf.transform(inp)
prediction=clf.predict(vect)
print(prediction)


# In[ ]:




