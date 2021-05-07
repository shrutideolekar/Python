# STAT4080 Data Programming with Python (online) - Project
# k nearest neighbours on the TunedIT data set

# Import packages
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import numpy.random as npr
from os import chdir, getcwd
from statistics import mode
from sklearn import metrics

wd=getcwd()
chdir(wd)

# For the project we will study the method of k nearest neighbours applied to a 
# music classification data set.These data come from the TunedIT website 
# http://tunedit.org/challenge/music-retrieval/genres
# Each row corresponds to a different sample of music from a certain genre.
# The original challenge was to classify the different genres (the original 
# prize for this was hard cash!). However we will just focus on a sample of the
# data (~4000 samples) which is either rock or not. There are 191 
# characteristics (go back to the website if you want to read about these)
# The general tasks of this exercise are to:
# - Load the data set
# - Standardise all the columns
# - Divide the data set up into a training and test set
# - Write a function which runs k nearest neighbours (kNN) on the data set.
#   (Don't worry you don't need to know anything about kNN)
# - Check which value of k produces the smallest misclassification rate on the 
#   training set
# - Predict on the test set and see how it does


# Q1 Load in the data using the pandas read_csv function. The last variable 
# 'RockOrNot' determines whether the music genre for that sample is rock or not
# What percentage of the songs in this data set are rock songs (to 1 d.p.)? 
data = pd.read_csv("tunedit_genres.csv")
rockPercent = data['RockOrNot'].value_counts(normalize=True) * 100
# Ans: 48.8 percent of the songs in the given data are rock songs.


# Q2 To perform a classification algorithm, you need to define a classification 
# variable and separate it from the other variables. We will use 'RockOrNot' as 
# our classification variable. Write a piece of code to separate the data into a 
# DataFrames X and a Series y, where X contains a standardised version of 
# everything except for the classification variable ('RockOrNot'), and y contains 
# only the classification variable. To standardise the variables in X, you need
# to subtract the mean and divide by the standard deviation

X = data.drop('RockOrNot',axis=1)
X = (X-X.mean())/X.std()
y = data['RockOrNot']

# Q3 Which variable in X has the largest correlation with y?
#get correlation of all variables in X with y
corr_xy = X.corrwith(y)

#get maximum correlation value either positive or negative
max_corr = max(corr_xy.abs())

#find the variable name with largest correlation with y
corr_xy[corr_xy  == max_corr].index.tolist()
corr_xy['PAR_SFM_M'] #to cross verify the correlation value

# Ans: 'PAR_SFM_M' has the largest correlation (positive) with y = 0.5962869174889944

# Q4 When performing a classification problem, you fit the model to a portion of 
# your data, and use the remaining data to determine how good the model fit was.
# Write a piece of code to divide X and y into training and test sets, use 75%
# of the data for training and keep 25% for testing. The data should be randomly
# selected, hence, you cannot simply take the first, say, 3000 rows. If you select 
# rows 1,4,7,8,13,... of X for your training set, you must also select rows 
# 1,4,7,8,13,... of y for training set. Additionally, the data in the training
# set cannot appear in the test set, and vice versa, so that when recombined,
# all data is accounted for. Use the seed 123 when generating random numbers
# Note: The data may not spilt equally into 75% and 25% portions. In this 
# situation you should round to the nearest integer. 

ind = list(range(X.shape[0])) 
trainSet = int(0.75 * X.shape[0]) # Train data is 75%, round to nearest integer -> 2995.25~2999
npr.seed(123) # Set seed
npr.shuffle(ind) # Shuffle the data as we want random data in train and test sets

# Set indices to split train and test sets
train_ind = ind[:trainSet]
test_ind = ind[trainSet:]

# split the actual data
X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

# Reset indices as shuffling of data results in shuffled unserialized indices
X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Q5 What is the percentage of rock songs in the training dataset and in the 
# test dataset? Are they the same as the value found in Q1?

rockPercentTrain = y_train.value_counts(normalize=True) * 100

rockPercentTest = y_test.value_counts(normalize=True) * 100

# Ans: The percentage of rock songs in the training dataset is 49.4 and that in the 
#test dataset is 47.1 which is closely equal to the value found in Q1 i.e 48.8%.

# Q6 Now we're going to write a function to run kNN on the data sets. kNN works 
# by the following algorithm:
# 1) Choose a value of k (usually odd)
# 2) For each observation, find its k closest neighbours
# 3) Take the majority vote (mean) of these neighbours
# 4) Classify observation based on majority vote

# We're going to use standard Euclidean distance to find the distance between 
# observations, defined as sqrt( (xi - xj)^T (xi-xj) )
# A useful short cut for this is the scipy functions pdist and squareform

# The function inputs are:
# - DataFrame X of explanatory variables 
# - binary Series y of classification values 
# - value of k (you can assume this is always an odd number)

# The function should produce:
# - Series y_star of predicted classification values

from scipy.spatial.distance import pdist, squareform
from math import inf

def kNN(X,y,k):
    # Find the number of obsvervation
    n = len(X)
    # Set up return values
    y_star = []
    # Calculate the distance matrix for the observations in X
    dist = squareform(pdist(X))
    # Make all the diagonals very large so it can't choose itself as a closest neighbour
    np.fill_diagonal(dist, inf)
    
    # Loop through each observation to create predictions
    for x in range(n):
        # Find the y values of the k nearest neighbours
        sortedDist = dist[x].argsort()[:k]
        y_nearest = []
        for ind in sortedDist:
            y_nearest.append(y[ind])
            
        # Now allocate to y_star        
        y_star.append(mode(y_nearest))
        
    return y_star

# Q7 The misclassification rate is the percentage of times the output of a 
# classifier doesn't match the classification value. Calculate the 
# misclassification rate of the kNN classifier for X_train and y_train, with k=3.
diff = (y_train - kNN(X_train,y_train,3)).abs()
mis_rate = round(diff.value_counts(normalize=True)[1] * 100,1)
# Ans: 4.7 (rounded to 1dp)
        
# Q8 The best choice for k depends on the data. Write a function kNN_select that 
# will run a kNN classification for a range of k values, and compute the 
# misclassification rate for each.

# The function inputs are:
# - DataFrame X of explanatory variables 
# - binary Series y of classification values 
# - a list of k values k_vals

# The function should produce:
# - a Series mis_class_rates, indexed by k, with the misclassification rates for 
# each k value in k_vals

def kNN_select(X,y,k_vals):
    n = len(X) # Find the number of obsvervations
    mis_class_rates = pd.Series([]) # Series to store misclassification rates
    dist = squareform(pdist(X)) # Calculate the distance matrix for the observations in X
    np.fill_diagonal(dist, inf) # Make all the diagonals very large so it can't choose itself as a closest neighbour
    
    # Loop through each k value to create predictions
    for k in k_vals:
        y_star = [] #reset y_star for every k
        
        # Loop through each observation to create predictions
        for x in range(n):
            # Find the y values of the k nearest neighbours
            sortedDist = dist[x].argsort()[:k]
            y_nearest = []
            for ind in sortedDist:
                y_nearest.append(y[ind])
                
            # Now allocate to y_star        
            y_star.append(mode(y_nearest))
         
        #Find misclassification rates for each value of k
        diff = (y - y_star).abs()
        mis_rate = round(diff.value_counts(normalize=True)[1] * 100,2)
        
        #Store all misclassification rates in a list
        mis_class_rates[k] = mis_rate
            
    return mis_class_rates

# Q9 Run the function kNN_select on the training data for k = [1, 3, 5, 7, 9] 
# and find the value of k with the best misclassification rate. Use the best 
# value of k to report the mis-classification rate for the test data. What is 
# the misclassification percentage with this k on the test set?

k_values = [1,3,5,7,9] # List of k values
mis_rates = kNN_select(X_train,y_train,k_values)
best_k_val = mis_rates.idxmin() #get the k value with lowest mis classification rate

# Misclassification percentage with best k on the test set
diff_test = (y_test - kNN(X_test,y_test,best_k_val)).abs()
mis_rate_test = round(diff_test.value_counts(normalize=True)[1] * 100,2)
print(mis_rate_test)

# Ans: The value of k with the best misclassification rate is 1 (3.33 is the misclassification rate)
# The misclassification percentage with the above obtained value of k on the test set is 5%.
# Note: Misclassification rates are rounded to 2 dp.

# Q10 Write a function to generalise the k nearest neighbours classification 
# algorithm. The function should:
# - Separate out the classification variable for the other variables in the dataset,
#   i.e. create X and y.
# - Divide X and y into training and test set, where the number in each is 
#   specified by 'percent_train'.
# - Run the k nearest neighbours classification on the training data, for a set 
#   of k values, computing the mis-classification rate for each k
# - Find the k that gives the lowest mis-classification rate for the training data,
#   and hence, the classification with the best fit to the data.
# - Use the best k value to run the k nearest neighbours classification on the test
#   data, and calculate the mis-classification rate
# The function should return the mis-classification rate for a k nearest neighbours
# classification on the test data, using the best k value for the training data
# You can call the functions from Q6 and Q8 inside this function, provided they 
# generalise, i.e. will work for any dataset, not just the TunedIT dataset.
def kNN_classification(df,class_column,seed,percent_train,k_vals):
    # df            - DataFrame to 
    # class_column  - column of df to be used as classification variable, should
    #                 specified as a string  
    # seed          - seed value for creating the training/test sets
    # percent_train - percentage of data to be used as training data
    # k_vals        - set of k values to be tests for best classification
    
    # Separate X and y
    X = df.drop(class_column,axis=1)
    X = (X-X.mean())/X.std()
    y = df[class_column]
    
    # Divide into training and test
    ind = list(range(X.shape[0])) 
    trainSet = int(percent_train * X.shape[0]) # Train data is 75%, round to nearest integer -> 2995.25~2999
    npr.seed(seed) # Set seed
    npr.shuffle(ind) # Shuffle the data as we want random data in train and test sets

    # Set indices to split train and test sets
    train_ind = ind[:trainSet]
    test_ind = ind[trainSet:]

    # split the actual data
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

    # Reset indices as shuffling of data results in shuffled unserialized indices
    X_train = X_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Run the k nearest neighbours classification on the training data
    n = len(X_train)
    mis_class_rates = pd.Series([]) # Series to store misclassification rates
    dist = squareform(pdist(X_train))  # Calculate the distance matrix for the observations in X
    np.fill_diagonal(dist, inf) # Make all the diagonals very large so it can't choose itself as a closest neighbour
    
    # Compute the mis-classification rates for each for the values in k_vals
    
    for k in k_vals: # Loop through each k value to create predictions
        y_star = [] #reset y_star for every k
        
        for x in range(n): # Loop through each observation to create predictions
            # Find the y values of the k nearest neighbours
            sortedDist = dist[x].argsort()[:k]
            y_nearest = []
            for ind in sortedDist:
                y_nearest.append(y_train[ind])
                
            y_star.append(mode(y_nearest)) # Now allocate to y_star
         
        #Find misclassification rates for each value of k
        diff = (y_train - y_star).abs()
        mis_rate = round(diff.value_counts(normalize=True)[1] * 100,2)
        
        mis_class_rates[k] = mis_rate #Store all misclassification rates in a list
        
    # Find the best k value, by finding the minimum entry of mis_class_rates 
    best_k = mis_class_rates.idxmin()
    
    # Run the classification on the test set to see how well the 'best fit'
    # classifier does on new data generated from the same source
    n_test = len(X_test)
    y_star_test = []
    dist_test = squareform(pdist(X_test)) # Calculate the distance matrix for the observations in X test set
    np.fill_diagonal(dist_test, inf) # Make all the diagonals very large so it can't choose itself as a closest neighbour
    
    # Loop through each observation to create predictions
    for x in range(n_test):
        # Find the y values of the k nearest neighbours
        sortedDist_test = dist_test[x].argsort()[:best_k]
        y_nearest_test = []
        for ind in sortedDist_test:
            y_nearest_test.append(y_test[ind])
                
        # y_star_test.append(round(Series(y_nearest_test).mean(),0))
        y_star_test.append(mode(y_nearest_test))
    
    # Calculate the mis-classification rates for the test data
    diff__test = (y_test - y_star_test).abs()
    mis_class_test = round(diff__test.value_counts(normalize=True)[1] * 100,2)

    return mis_class_test
    
# Test your function with the TunedIT data set, with class_column = 'RockOrNot',
# seed = the value from Q4, percent_train = 0.75, and k_vals = set of k values
# from Q8, and confirm that it gives the same answer as Q9.
print(kNN_classification(data,'RockOrNot',123,0.75,k_values))

#Note: I hae printed the mis classification rates from Q9 and Q10 to cross verify that both give same answers.

# Now test your function with another dataset, to ensure that your code 
# generalises. You can use the house_votes.csv dataset, with 'Party' as the 
# classifier. Select the other parameters as you wish.
# This dataset contains the voting records of 435 congressman and women in the 
# US House of Representatives. The parties are specified as 1 for democrat and 0
# for republican, and the votes are labelled as 1 for yes, -1 for no and 0 for
# abstained.
# Your kNN classifier should return a mis-classification for the test data (with 
# the best fit k value) of ~8%.
dataHouse = pd.read_csv("house_votes.csv")
print(kNN_classification(dataHouse,'Party',123,0.75,k_values))

#Ans: Mis-classification for the test data (with the best fit k value) is 8.26% (rounded to 2dp)
