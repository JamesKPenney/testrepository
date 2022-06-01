#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>
# 

# In this notebook we try to practice all the classification algorithms that we have learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[2]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object
# 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[9]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[10]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[11]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[12]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[13]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[14]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[15]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[16]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[17]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[18]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[19]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[20]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[21]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size = 0.2)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 12
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    neighbr = KNeighborsClassifier(n_neighbors = n).fit(Xtrain, Ytrain)
    yhat = neighbr.predict(Xtest)
    mean_acc[n-1] = metrics.accuracy_score(Ytest, yhat)
    std_acc[n-1] = np.std(yhat == Ytest)/np.sqrt(yhat.shape[0])
mean_acc
#Unsure how to resolve warning for np.int deprecation
#Can this lead to rounding errors and incorrect output?


# In[26]:


plt.plot(range(1,Ks),mean_acc)
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy', '+/- std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy is:", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

neigh = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1).fit(Xtrain, Ytrain)


# # Decision Tree
# 

# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
from sklearn import tree

modeltree = DecisionTreeClassifier(criterion="entropy")
modeltree.fit(Xtrain, Ytrain)
predTree = modeltree.predict(Xtest)
print(predTree[0:5])
print(Ytest[0:5])


# In[28]:


get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydotplus')
import graphviz 
import pydotplus

dotdata = StringIO()
filename = "tree.png"
names = Feature.columns
out=tree.export_graphviz(modeltree,feature_names=names, out_file=dotdata, class_names= np.unique(Ytrain), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dotdata.getvalue())  
graph = pydotplus.graph_from_dot_data(dotdata.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:





# # Support Vector Machine
# 

# In[30]:


import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn import svm

classf = svm.SVC(kernel="sigmoid")
classf.fit(Xtrain, Ytrain)


# In[31]:


yhat = classf.predict(Xtest)


# In[32]:


from sklearn.metrics import f1_score
f1_score(Ytest,yhat, average="weighted")


# # Logistic Regression
# 

# In[33]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver="liblinear").fit(Xtrain,Ytrain)


# In[35]:


yhat = LR.predict(Xtest)
yhat_prob = LR.predict_proba(Xtest)


# In[46]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(Ytest,yhat)


# # Model Evaluation using Test set
# 

# In[38]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:
# 

# In[39]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation
# 

# In[40]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[48]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_feature = test_df[['Principal','terms','age','Gender','weekend']]
test_feature = pd.concat([test_feature,pd.get_dummies(test_df['education'])], axis=1)
test_feature.drop(['Master or Above'], axis = 1,inplace=True)
test_feature.head()

test_X = preprocessing.StandardScaler().fit(test_feature).transform(test_feature)
test_X[0:5]

test_y = test_df['loan_status'].values
test_y[0:5]


# In[49]:


TestX = preprocessing.StandardScaler().fit(test_feature).transform(test_feature)
TestX[0:5]
TestY = test_df['loan_status'].values
TestY[0:5]


# In[50]:


knn_pred=neigh.predict(TestX)
jc1=jaccard_similarity_score(TestY, knn_pred)
fs1=f1_score(TestY, knn_pred, average='weighted')

tree_pred=modeltree.predict(TestX)
jc2=jaccard_similarity_score(TestY, tree_pred)
fs2=f1_score(TestY, tree_pred, average='weighted')

svm_pred=clf.predict(TestX)
jc3=jaccard_similarity_score(TestY, svm_pred)
fs3=f1_score(TestY, svm_pred, average='weighted')

log_pred=LR.predict(TestX)
proba=LR.predict_proba(TestX)
jc4=jaccard_similarity_score(TestY, log_pred)
fs4=f1_score(TestY, log_pred, average='weighted')  
ll4=log_loss(TestY, proba)

list_jc = [jc1, jc2, jc3, jc4]
list_fs = [fs1, fs2, fs3, fs4]
list_ll = ['NA', 'NA', 'NA', ll4]


import pandas as pd

df = pd.DataFrame(list_jc, index=['KNN','Decision Tree','SVM','Logistic Regression'])
df.columns = ['Jaccard']
df.insert(loc=1, column='F1-score', value=list_fs)
df.insert(loc=2, column='LogLoss', value=list_ll)
df.columns.name = 'Algorithm'
df.head()


# # Report
# 
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# | Algorithm          | Jaccard | F1-score | LogLoss |
# | ------------------ | ------- | -------- | ------- |
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |
# 

# In[51]:


df.head()


# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                                                             |
# | ----------------- | ------- | ------------- | ------------------------------------------------------------------------------ |
# | 2020-10-27        | 2.1     | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |
# | 2020-08-27        | 2.0     | Malika Singla | Added lab to GitLab                                                            |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <p>
# 
