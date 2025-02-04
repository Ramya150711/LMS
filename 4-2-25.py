#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install emot')


# In[8]:


text1="What are you saying 😂.I am the boss 😎,and why are you so 😏"


# In[9]:


import re
from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO


# In[10]:


# Function for converting emojis into word
def converting_emojis(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text
converting_emojis(text1)


# In[11]:


def emoji_removal(string):
    emoji_unicodes = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols 
                               u"\U0001F680-\U0001F6FF"  # transport 
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_unicodes.sub(r'', string)
emoji_removal(text1)


# In[ ]:




