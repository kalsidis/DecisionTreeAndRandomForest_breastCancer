# =============================================================================
# HOMEWORK 2 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: gliapisa@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'tree' package, for creating the DecisionTreeClassifier and using graphviz
# 'model_selection' package, which will help test our model.
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets,metrics,tree,model_selection
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Load breastCancer data
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
breastCancer =datasets.load_breast_cancer()

df = pd.DataFrame(breastCancer.data, columns = breastCancer.feature_names)
df['target'] = breastCancer.target

# =============================================================================

# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:,:numberOfFeatures]
Y = breastCancer.target
print(len(breastCancer.feature_names))
print(breastCancer.feature_names)

#ΟΠΩΣ ΤΟ ΕΚΑΝΑ ΣΤΗΝ ΠΡΟΗΓΟΥΜΕΝΗ ΕΡΓΑΣΙΑ ΜΕ ΤΟ iloc
#ΒΓΑΖΕΙ ΕΝΑ ERROR ΟΤΙ ΔΕΝ ΔΕΧΕΤΑΙ ΣΥΝΕΧΗΣ ΜΕΤΑΒΛΗΤΗ CONTINOUS
#x=df.iloc[:,0:10]
#y=df.iloc[:,10]
#print(x)
#print(len(x))
#print(y)
#print(len(y))

X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size = 0.25, random_state = 0)

# DecisionTreeClassifier() is the core of this script. You can customize its functionality
# in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the information gain.
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================


# ADD COMMAND TO CREATE DECISION TREE CLASSIFIER MODEL HERE

clf=DecisionTreeClassifier(criterion='gini',max_depth=3)
model=clf.fit(X_train,Y_train)
Y_predicted=clf.predict(X_test)



#Με τα x μικρό και το y μικρό
#clf=DecisionTreeClassifier()
#model=clf.fit(x_train, y_train)
#y_predicted = clf.predict(x_test)

# =============================================================================

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.

# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE

# =============================================================================

# Ok, now let's predict the output for the test input set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE
# =============================================================================

# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'accuracy_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Accuracy: %2f" % metrics.accuracy_score(Y_test, Y_predicted))
print("Recall: %2f" % metrics.recall_score(Y_test, Y_predicted, average="macro"))
print("Precision: %2f" % metrics.precision_score(Y_test, Y_predicted, average="macro"))
print("F1: %2f" % metrics.f1_score(Y_test, Y_predicted, average="macro"))

# =============================================================================

#print(tree.export_text(clf))#αυτό φτιάχνει το δέντρο κάτω αλλά σε μια άσχημη και δυσνόητη μορφή
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(30,80))
tree.plot_tree(clf,feature_names = breastCancer.feature_names[:numberOfFeatures],class_names = breastCancer.target_names,filled = True)
plt.show()
# By using the 'plot_tree' function from the tree classifier we can visualize the trained model.
# There is a variety of parameters to configure, which can lead to a quite visually pleasant result.
# Make sure that you set the following parameters within the function:

# ============================================================================