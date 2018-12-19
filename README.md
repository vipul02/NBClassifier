# NBClassifier
Naive Bayes Classifier from scratch in python3

# **Naive Bayes Classifier For Classifying Whether The Tumor Is Benign or Malignant**
***

**What is Naive Bayes algorithm?**

Naive Bayes is a classification technique based on Bayes’ Theorem(*Probability theory*) with an assumption that all the features that predicts the target value are independent of each other. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature in determining the target value.

> Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

Bayes theorem provides a way of calculating posterior probability P(c|x) - *(read as Probability of **c** given **x**)*,  from P(c), P(x) and P(x|c). Look at the equation below:
>
> $\mathbf{P} \left({x \mid c} \right) = \frac{\mathbf{P} \left ({c \mid x} \right) \mathbf{P} \left({c} \right)}{\mathbf{P} \left( {x} \right)}$

where,

* *x is set of features*
* *c is set of classes*
* P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
* P(c) is the prior probability of class **c**.
* P(x|c) is the observation density or likelihood which is the probability of predictor(the query  **x**) given class.
* P(x) is the prior probability of predictor **x**, and it is also called as Evidence.

**Why should we use Naive Bayes ?**

* As stated above, It is **_easy_** to build and is particularly useful for **_very large data sets_**.
* It is **extremely fast** for both training and prediction.
* It provide straightforward probabilistic prediction.
* It is often very easily interpretable.
* It has very few (if any) tunable parameters.
* It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).


```python
#Importing the Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random 
import scipy.stats as S
```


```python
# Importing tha Datasets

Data = pd.read_csv("data.csv")
Data.dropna(axis=1,inplace=True)
```


```python
Data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



# Principal Component Analysis(PCA)

it is generally used for dimensionality reduction...
In this data we are using PCA for finding important features which have more effect/weightage on finding posterior probability


```python
def PCA(Data):
    Data_array = np.array(Data)

    #______ Make Zero Mean Distribution_____

    Means_value = np.mean(Data_array,axis=0)  #finding mean of each columns
    Means_value = Means_value.reshape(1,Data_array.shape[1])


    Centred_value = Data_array - Means_value    #Substracting respective mean with their respective columns values

    
    #___finding covariance matrix of zero mean distrubuted value____

    Covariance_matrix = np.cov(Centred_value , rowvar=0)    
    Covariance_matrix.shape
    
    #__finding eigen values and eigen vectors of covariance matrix
    values,vectors = np.linalg.eig(Covariance_matrix)
    values = values.reshape(1,len(values))

    values_index = np.argsort(values)   #getting original index on the basis of sorted values
    values_index = values_index[0]


    values_index = (values_index[::-1])   # transform sorted_index to descn. order
    
    values = values[:,values_index]     #getting values which will be in descn. order
    
    
    #_______finding cummulative sum for calculating weightage change____
    weightage_of_features = np.cumsum(values)/np.sum(values)
    
    features_list_index=[]  #__list of important features index

    for i in range(0,len(weightage_of_features)):
        weightage_in_percent = weightage_of_features[i]*100

        if weightage_in_percent <= 99.9:
            features_list_index.append(values_index[i])

    return(features_list_index)  
```


```python
no_features = PCA(Data.iloc[:, 2:])
print(no_features)
```

    [0, 1]
    

> By Applying PCA on our dataset, important features are 2+0, 2+1 i.e. 2, 3 index of our data which is radius_mean and texture_mean

> *Now split our Data into training and testing set*


```python
train, test = train_test_split(Data, test_size=0.3)
```

># Training of Model and Seperating By Class: { Benign, Malignant }


```python
# Seperarting data by class
dataB = train[train['diagnosis'] == 'B']
dataM = train[train['diagnosis'] == 'M']

# This function returns the mean and covariance matrix of provided data
def calculate_mean_covMat(data):
    return data.iloc[:, 2:4].mean(), np.cov(data.iloc[:, 2:4],rowvar=0)

# Calculating mean and covariance matrix of Benign
BT_mean, BT_cov = calculate_mean_covMat(dataB)

# Calculating mean and covariance matrix of Benign
MT_mean, MT_cov = calculate_mean_covMat(dataM)

# Calculating the P(B) and P(M) independently
P_B = dataB.shape[0]/train.shape[0]
P_M = dataM.shape[0]/train.shape[0]
```

Before we go any further we should know **Posterior Conditional Probability** which is,

$\mathbf{P} \left({x \mid c} \right) = \mathbf{P} \left ({c \mid x} \right) \mathbf{P} \left({c} \right)$

where, $\mathbf{P} \left ({c \mid x} \right)$ is ***Observation Distribution***

And Mathematical Formula of Observation Distribution is

$$ \frac {1}{(\sqrt{2}\pi)^2\sqrt{\textstyle\sum}}e^{-0.5}A^T{\textstyle\sum}^{-1}A $$

where,

* $ \textstyle\sum $    is a covariance matrix

* A is a vector which contains 
$
A=
  \left [ 
      {\begin{array}{c}
           R_i - Mean(radius\_mean) \\
           T_i - Mean(texture\_mean) \\
      \end{array} } 
  \right]
$

># Testing of Model


```python
# This function returns the Observation Distribution
def calculateObservationDistribution(test, mean, covMat):
    return S.multivariate_normal.pdf(test, mean, covMat)


# Here we are Calculating the Posterior Conditional Probability of Benign and Malignant Data
PosteriorConditionalProbabilityB = calculateObservationDistribution(test.iloc[:, 2:4], BT_mean, BT_cov)*P_B
PosteriorConditionalProbabilityM = calculateObservationDistribution(test.iloc[:, 2:4], MT_mean, MT_cov)*P_M
```

># ***Prediction***


```python
# In this section we are labelling whether it is Benign or Malignant

# creating empty list of label prediction
label_prediction = []

# Comparing PosteriorConditionalProbability of Benign and Malignant
for b, m in zip(range(len(PosteriorConditionalProbabilityB)), range(len(PosteriorConditionalProbabilityM))):
    if(PosteriorConditionalProbabilityB[b] > PosteriorConditionalProbabilityM[m]):
        label_prediction.append('B')
    else:
        label_prediction.append('M')

# list to array
label_prediction = np.array(label_prediction)
```

> # ***Finding an Accuracy***


```python
# Comapring all the rows of diagnosis of test data with label prediction
count = 0
total = len(test)
for i in range(total):
    if test.iloc[i, 1] == label_prediction[i]:
        count += 1
accuracy = count/total
print('Accuracy = ' + str(accuracy*100) + '%')
```

    Accuracy = 91.22807017543859%
    
