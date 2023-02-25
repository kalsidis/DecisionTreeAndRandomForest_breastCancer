# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: gliapisa@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,metrics,tree,model_selection,ensemble
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

# =============================================================================

# Load breastCancer data
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
breastCancer =datasets.load_breast_cancer()
print(breastCancer.feature_names)

# =============================================================================

# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.

numberOfFeatures = 10
X = breastCancer.data
Y = breastCancer.target

#x=df.iloc[:,:30]
#y=df.iloc[:,10]
#print(x)

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Also, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.

X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.25,random_state=0)
#x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25,random_state=0)

# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================

# ADD COMMAND TO CREATE RANDOM FOREST CLASSIFIER MODEL HERE
CLF = ensemble.RandomForestClassifier(criterion='gini', n_estimators=50, max_depth=3)
MODEL = CLF.fit(X_train, Y_train)
Y_predicted = CLF.predict(X_test)

A=[]
for i in range(1,201):
    CLF=ensemble.RandomForestClassifier(criterion='gini',n_estimators=i,max_depth=3)
    MODEL = CLF.fit(X_train, Y_train)
    Y_predicted = CLF.predict(X_test)
    A.append(metrics.f1_score(Y_test, Y_predicted, average="macro"))
print(A)
a=[]
for i in range(1,201):
    a.append(i)

plt.scatter(a,A)
plt.plot(a,A)
plt.grid()
plt.show()
#clf=ensemble.RandomForestClassifier(criterion='gini',n_estimators=10,max_depth=3)
#model=clf.fit(x_train,y_train)
#y_predicted=clf.predict(x_test)
# =============================================================================

# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN YOUR MODEL HERE

# =============================================================================

# Ok, now let's predict the output for the test set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE

# =============================================================================

# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used (for now):
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
#             This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)

print("Accuracy: %2f" % metrics.accuracy_score(Y_test,Y_predicted))
print("Recall: %2f" % metrics.recall_score(Y_test,Y_predicted, average="macro"))
print("Precision: %2f" % metrics.precision_score(Y_test, Y_predicted, average="macro"))
print("F1: %2f" % metrics.f1_score(Y_test, Y_predicted, average="macro"))
# =============================================================================

# A Random Forest has been trained now, but let's train more models, 
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# CREATE MODELS AND PLOTS HERE



#for i in range(200):
#    numberOfFeatures = 10
#    X = breastCancer.data[:, :numberOfFeatures]
#    y = breastCancer.target

#    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#    clf = ensemble.RandomForestClassifier(criterion='gini', n_estimators=10, max_depth=3)
#    model = clf.fit(X_train, Y_train)
#    Y_predicted = clf.predict(X_test)

#=============================================================================