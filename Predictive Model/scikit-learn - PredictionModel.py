#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('E:\pythonprogs\pythonrepo\Predictive Model')


# In[5]:


os.getcwd()


# In[6]:


import pandas as pd
import numpy as np


# In[7]:


dataset = pd.read_csv('iris.csv')


# In[8]:


dataset


# In[13]:


x = dataset.iloc[:,0:4].values


# In[16]:


y = dataset.iloc[:,4]


# In[18]:


y


# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)


# In[24]:


y


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[29]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[30]:


y_pred = logmodel.predict(x_test)


# In[36]:


y_pred
y_test


# In[32]:


from sklearn.metrics import confusion_matrix


# In[34]:


confusion_matrix(y_test,y_pred)


# In[37]:


28/30


# In[42]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier_knn.fit(x_train,y_train)


# In[44]:


y_pred = classifier_knn.predict(x_test)


# In[45]:


confusion_matrix(y_test,y_pred)


# In[46]:


29/30


# In[49]:


from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train,y_train)


# In[51]:


y_pred = classifier_nb.predict(x_test)


# In[53]:


confusion_matrix(y_test,y_pred)


# In[54]:


28/30


# In[55]:


from sklearn.svm import SVC
classifier_svm_sigmoid = SVC(kernel='sigmoid')
classifier_svm_sigmoid.fit(x_train,y_train)


# In[59]:


y_pred = classifier_svm_sigmoid.predict(x_test)


# In[60]:


confusion_matrix(y_test,y_pred)


# In[62]:


from sklearn.svm import SVC
classifier_svm_linear = SVC(kernel = 'linear')
classifier_svm_linear.fit(x_train,y_train)


# In[63]:


y_pred = classifier_svm_linear.predict(x_test)


# In[67]:


confusion_matrix(y_test,y_pred)


# In[68]:


29/30


# In[69]:


from sklearn.svm import SVC
classifier_svm_rbf = SVC(kernel='rbf')
classifier_svm_rbf.fit(x_train,y_train)


# In[70]:


y_pred = classifier_svm_rbf.predict(x_test)


# In[71]:


confusion_matrix(y_test,y_pred)


# In[72]:


from sklearn.svm import SVC
classifier_svm_poly = SVC(kernel='poly')
classifier_svm_poly.fit(x_train,y_train)


# In[73]:


y_pred = classifier_svm_poly.predict(x_test)


# In[74]:


confusion_matrix(y_test,y_pred)


# In[76]:


from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy')
classifier_dt.fit(x_train,y_train)


# In[77]:


y_pred = classifier_dt.predict(x_test)


# In[78]:


confusion_matrix(y_pred,y_test)


# In[79]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=3,criterion='entropy')
classifier_rf.fit(x_train,y_train)


# In[81]:


y_pred = classifier_rf.predict(x_test)


# In[82]:


(/confusion_matrix(y_test,y_pred))


# In[83]:


28/30


# In[ ]:




