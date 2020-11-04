# Introduction to Applied Machine learning using python


## Examples of machine learning problems

- 1) Credit card fraud detection
- 2) Movie recommendations
- 3) speech recognition
- 4) Face recognition
- 5) Medical dignosis
- 6) Business usecases

---

## Supervised Learning

Supervised learning needs to have this training set with labeled objects in order to make its predictions.
Our goal in the ***Supervised Learning*** is to learn some function that maps data item in X to a label in Y


### 1) Classification

    If the output is a category a finite number of possibilities such as a fraudulent or not fraudulent prediction for a credit card transaction.
    Or maybe it's the English word associated with an audio signal for speech recognition. 
    We call this a classification problem within supervised learning, and the function that we learn is called the classifier.


### 2) Regression 

    If the output variable we want to predict is not a category, but a real valued number like the amount of time in seconds 
    it would likely take a car to accelerate from 0 to 100 kilometers per hour. We call that regression problem. 
    And we're learning something called a regression function

---

## Unsupervised learning

In many cases we only have input data, we don't have any labels to go with the data. And in those cases the problems we can solve involve taking the input data and trying to find some kind of useful structure in it.

So once we can discover this structure in the form of clusters, groups or other interesting subsets. The structure can be used for tasks like producing a useful summary of the input data maybe visualizing the structure.

---

## Basic Machine learning workflow

`Representation ==> Evaluation ==> Optimization`

### 1- Representation

- **Do:**
    - Convert the data samples into set of features.
    - Choose A Feature Representation (an image as an array of pixels for example).
    - Do Feature Engineering and Extraction. 
    - Choose A learning classifier to use.

#### Why is it important to examine your dataset before starting to work with it for a machine learning task?

- To understand how much missing data there is in the dataset
- To notice if we have enough examples for each class.
- To check for existance of noisy data or any inconsistancy in features.
- To get an idea of whether the features need further cleaning
- It may turn out the problem doesn't require machine learning.

#### Why is it important to visualize your dataset?

- we can get an idea of the range of values that each feature takes on.
- we can immediately see any unusual outliers 
- we can se how well clustered and well separated the different types of objects are in feature space. 


#### K-Nearest Neighbors Classification

- 1. A distance metric (the euclidean metric, Minkowski metric)
- 2. How many 'nearest' neighbors to look at? (k=?)
- 3. Optional weighting function on the neighbor points.
    - For example, we may decide that neighbors that are closer to the new instance that we're trying to classify, should have more influence, or more votes on the final label.
- 4. Methods for aggregating the classes of neighbor points (simple majority votes).

     when K = 1, the prediction is sensitive to noise, outliers, mislabeled data, and other sources of variation in individual data points.
     
     For larger values of K, the areas assigned to different classes are smoother and not as fragmented, and more robust to noise in the individual points. But possibly with some mistakes, more mistakes in individual points.
     
     In general with k-nearest neighbors, using a larger k suppresses the effects of noisy individual labels. But results in classification boundaries that are less detailed.


---

## 2- Train a classifier

fit a classifier on a training dataset.

---
## 3- Evaluation 

- **Choose:**
    -  What criterion distinguishes good vs. bad classifiers?

---
## 4- Optimization

- **Choose:**
    - How to search for the ***settings/parameters*** that give the best classifier for this evaluation crietrion.


---
3- What’s new for you ?
-

- `pandas.plotting.scatter_matrix(frame, alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwargs)`
- Deep understanding of bias


4- Resources ? 
-
- [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html)
- [http://approximatelycorrect.com/2016/11/07/the-foundations-of-algorithmic-bias/](http://approximatelycorrect.com/2016/11/07/the-foundations-of-algorithmic-bias/)