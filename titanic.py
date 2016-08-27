import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from sklearn.cross_validation import train_test_split

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

train = pandas.read_csv('data/titanic_train.csv')
y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(accuracy_score(lr.predict(X_test), y_test))


# Linear classifier.

random.seed(42)
tflr = learn.LinearClassifier(n_classes=2,
    feature_columns=learn.infer_real_valued_columns_from_input(X_train),
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
tflr.fit(X_train, y_train, batch_size=128, steps=500)
print(accuracy_score(tflr.predict(X_test), y_test))

# 3 layer neural network with rectified linear activation.

random.seed(42)
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10],
    n_classes=2,
    feature_columns=learn.infer_real_valued_columns_from_input(X_train),
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
classifier.fit(X_train, y_train, batch_size=128, steps=500)
print(accuracy_score(classifier.predict(X_test), y_test))

# 3 layer neural network with hyperbolic tangent activation.

def dnn_tanh(features, target):
    target = tf.one_hot(target, 2, 1.0, 0.0)
    logits = layers.stack(features, layers.fully_connected, [10, 20, 10],
        activation_fn=tf.tanh)
    prediction, loss = learn.models.logistic_regression(logits, target)
    train_op = layers.optimize_loss(loss,
        tf.contrib.framework.get_global_step(), optimizer='SGD', learning_rate=0.05)
    return tf.argmax(prediction, dimension=1), loss, train_op

random.seed(42)
classifier = learn.Estimator(model_fn=dnn_tanh)
classifier.fit(X_train, y_train, batch_size=128, steps=100)
print(accuracy_score(classifier.predict(X_test), y_test))

