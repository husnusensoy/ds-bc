#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


r = np.array([[7, 6, 7, 4, 5, 4],
              [6, 7, np.nan, 4, 3, 4],
              [np.nan, 3, 3, 1, 1, np.nan],
              [1, 2, 3, 3, 3, 4],
              [1, np.nan, 1, 2, 3, 3]])

r


# In[3]:


m, _ = r.shape
s = np.zeros((m, m))


# In[4]:


s


# In[7]:


mu = np.nanmean(r,axis=1) #np.mean(r,axis=1)


# In[8]:


mu


# In[9]:


i,j = 1,2


# In[14]:


mask  = ~np.isnan(r[i,:]) & ~np.isnan(r[j,:])


# In[22]:


num = np.dot(r[i,mask] - mu[i], r[j,mask] - mu[j])


# In[23]:


denum = np.linalg.norm(r[i,mask] - mu[i]) *  np.linalg.norm(r[j,mask] - mu[j])


# In[25]:


num/denum


# In[29]:


def sim(r:np.ndarray) -> np.ndarray:
    m,_ = r.shape
    
    s = np.zeros((m,m))
    
    mu = np.nanmean(r,axis=1) 
    
    for i in range(m):
        for j in range(m):
            mask  = ~np.isnan(r[i,:]) & ~np.isnan(r[j,:])
            
            num = np.dot(r[i,mask] - mu[i], r[j,mask] - mu[j])
            
            denum = np.linalg.norm(r[i,mask] - mu[i]) *  np.linalg.norm(r[j,mask] - mu[j])
            
            s[i][j] = num/denum
            
    return s
            


# In[30]:


s = sim(r)


# In[31]:


s


# In[36]:


~np.isnan(r[:,2])


# In[42]:


mask =  np.nonzero(~np.isnan(r[:,2]))[0]


# In[43]:


mask


# In[52]:


mask[(-s[1,mask]).argsort()]


# In[48]:


s[1,mask]


# In[53]:


r[[0,2],2] 


# In[54]:


mu = np.nanmean(r,axis=1) 


# In[59]:


zero_centered_score = r[[0,2],2]  - mu[[0,2]]


# In[58]:


weight = s[1,[0,2]]


# In[60]:


zero_centered_score, weight


# In[63]:


np.dot(zero_centered_score,weight)/np.abs(weight).sum() + mu[1]


# In[65]:


import pandas as pd

df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data',delimiter=r'\t',names=['user_id', 'item_id','rating','timestamp'])

X = df.pivot(index='user_id', columns='item_id', values='rating').values


# In[67]:


X.shape


# In[ ]:





# In[ ]:




