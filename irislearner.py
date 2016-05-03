# (SUPERVISED LEARNING) IT'S NOT HARD

# 1: COLLECT/PARSE DATA

# 1a: Import sklearn + tensorflow py libs
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris() # STOCK DATASET
# features: petalx, petaly, sepalx, sepaly

# 2: Decide how to Discretize data.
# In this case, we already know that there are 3 flower types
# swap the thing after skflow. and it's parameter to seap the model chosen

# deep neural nets
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
# hidden units pertain to the hidden neural net layers. More time for big numbers I guess.

# or we could've done simple lin. regression
# ...   .TensorFlowLinearClassifier(n_classes=3)

# 3: Train Data
# We use a training alg. called "fit"
classifier.fit(iris.data, iris.target)

# 4: Reveal results

score = metrics.accuracy_score(iris.target,classifier.predict(iris.data))
print("Accuracy: %f" % score)
