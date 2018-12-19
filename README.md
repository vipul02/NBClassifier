# NBClassifier
Naive Bayes Classifier from scratch in python3

# **Naive Bayes Classifier For Classifying Whether The Tumor Is Benign or Malignant**
***

**What is Naive Bayes algorithm?**

Naive Bayes is a classification technique based on Bayes’ Theorem(*Probability theory*) with an assumption that all the features that predicts the target value are independent of each other. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature in determining the target value.

> Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

A custom implementation of a Naive Bayes Classifier written from scratch in Python 3.

[![Bayes Theorem](bayes-theorem.png)](http://www.saedsayad.com/naive_bayesian.htm)

From [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier):

> In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.


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


# Principal Component Analysis(PCA)

it is generally used for dimensionality reduction...
In this data we are using PCA for finding important features which have more effect/weightage on finding posterior probability



> By Applying PCA on our dataset, important features are 2+0, 2+1 i.e. 2, 3 index of our data which is radius_mean and texture_mean

> *Now split our Data into training and testing set*


># Training of Model and Seperating By Class: { Benign, Malignant }



**Posterior Conditional Probability** ,

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

