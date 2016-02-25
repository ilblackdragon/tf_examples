import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from sklearn.cross_validation import train_test_split

import tensorflow as tf

import skflow

train = pandas.read_csv('data/titanic_train.csv')
y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(accuracy_score(lr.predict(X_test), y_test))


# Linear classifier.

random.seed(42)
tflr = skflow.TensorFlowLinearClassifier(n_classes=2, batch_size=128,
                                         steps=500, learning_rate=0.05)
tflr.fit(X_train, y_train)
print(accuracy_score(tflr.predict(X_test), y_test))

# 3 layer neural network with rectified linear activation.

random.seed(42)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
    n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
classifier.fit(X_train, y_train)
print(accuracy_score(classifier.predict(X_test), y_test))

# 3 layer neural network with hyperbolic tangent activation.

def dnn_tanh(X, y):
    layers = skflow.ops.dnn(X, [10, 20, 10], tf.tanh)
    return skflow.models.logistic_regression(layers, y)

random.seed(42)
classifier = skflow.TensorFlowEstimator(model_fn=dnn_tanh,
    n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
classifier.fit(X_train, y_train)
print(accuracy_score(classifier.predict(X_test), y_test))
