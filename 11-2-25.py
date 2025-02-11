#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install pytextrank')


# In[2]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[5]:


import spacy
import pytextrank


# In[6]:


document="""Not only did it only confirm that the film would be unfunny and generic,but it also managed to give away the entire movie;
and I'm not exaggerating -every moment,every plot point,every joke is told in the trailer
"""


# In[7]:


en_nlp=spacy.load("en_core_web_sm")
en_nlp.add_pipe("textrank")
doc=en_nlp(document)


# In[8]:


tr=doc._.textrank
print(tr.elapsed_time)


# In[10]:


for combination in doc._.phrases:
    print(combination.text,combination.rank,combination.count)


# In[11]:


from bs4 import BeautifulSoup
from urllib.request import urlopen


# In[16]:


def get_only_text(url):
    page=urlopen(url)
    soup=BeautifulSoup(page)
    text='\t'.join(map(lambda p: p.text,soup.find_all('p')))
    print(text)
    return soup.title.text,text


# In[17]:


url="https://en.wikipedia.org/wiki/Natural_language_processing"
text=get_only_text(url)


# In[18]:


len(''.join(text))


# In[19]:


text[:1000]


# In[20]:


get_ipython().system('pip install sumy')


# In[21]:


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer


# In[25]:


LANGUAGE="english"
SENTENCES_COUNT=10
url="https://en.wikipedia.org/wiki/Natural_language_processing"
parser=HtmlParser.from_url(url,Tokenizer(LANGUAGE))
summarizer=LsaSummarizer()
sumarizer=LsaSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words=get_stop_words(LANGUAGE)
for sentence in summarizer(parser.document,SENTENCES_COUNT):
    print(sentence)


# In[26]:


text="""Not only did it only confirm that the film would be unfunny and generic,but it also managed to give away the entire movie;
and I'm not exaggerating -every moment,every plot point,every joke is told in the trailer
"""


# In[27]:


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer


# In[28]:


parser=PlaintextParser.from_string(text,Tokenizer("english"))


# In[29]:


from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words
summarizer_lex=LexRankSummarizer()
summarizer_lex.top_words=get_stop_words("english")


# In[ ]:




