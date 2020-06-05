#!/usr/bin/env python
# coding: utf-8

# ## Vocabulary expansion using word embeddings model
# 
# It is related to a task of identifying alcoholic items in notes,
# just a first step in direction to better perform this task.
# 
# There is a previous list of names (types and brands) here: https://docs.google.com/document/d/1yigf9iJ6FDEj_nGUThH48r2IwAww9pQQdlwhAf7Hg1U/edit
# 
# In the next cells we search for more words which are similar with these seeds.
# 
# The result of this method needs a human validation.

# In[1]:


import requests as req
from bs4 import BeautifulSoup
import json


# Using sets of words from this gdocs

# In[2]:


resp_body = req.get('https://docs.google.com/document/d/1yigf9iJ6FDEj_nGUThH48r2IwAww9pQQdlwhAf7Hg1U/edit')


# In[3]:


soup = BeautifulSoup(resp_body.text, 'html.parser')
title = soup.title.string
l = [i for i in soup.find_all(type='text/javascript') if i.string.find('DOCS_modelChunk =') != -1]

content = l[0].contents[0].string
content = content.replace('DOCS_modelChunk = ', '')
content = content[:content.find(']') + 1]
content_dict = json.loads(content)


# In[4]:


text = content_dict[0]['s']
doc_lists = list(filter(lambda l: len(l) > 0, text.split('\n')))
doc_lists = [l.replace('\t', '') for l in doc_lists]


# Here we are considering that:
# - lowercase words are drink categories
# - otherwise, if uppercase in first letter, then it is a brand

# In[5]:


brands = set()
drinks = set()
for line in doc_lists:
    entities = [e.strip() for e in line.split('=')[1].split(',')]
    for e in list(filter(str.islower, entities)):
        drinks.add(e)
    for e in list(filter(lambda e: e[0].isupper(), entities)):
        brands.add(e)


# We can expand this vocabulary by finding nearest neighbor words using a word embeddings model (portuguese corpus)

# In[6]:


import os.path
from gensim.models import KeyedVectors

if not os.path.isfile('/tmp/skip_s50.txt'):
    get_ipython().system("wget --no-check-certificate         'http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s50.zip'         -O /tmp/fasttext_skip_s50.zip")

get_ipython().system('unzip -qq -n /tmp/fasttext_skip_s50.zip -d /tmp')

model = KeyedVectors.load_word2vec_format('/tmp/skip_s50.txt')


# In[31]:


# TODO: add to google-doc this drink type: "saquê"
drinks.add('saquê')

drinks_exp = {}
for d in drinks:
    drinks_exp[d] = []
    if d in model:
        drinks_exp[d] += [w[0] for w in model.most_similar(d) if not w[0] in drinks]

drinks_exp

