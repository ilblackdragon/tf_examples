import random
from sklearn import datasets, cross_validation, metrics
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

random.seed(42)

# Load dataset and split it into train / test subsets.

digits = datasets.load_digits()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=42)

# TensorFlow model using Scikit Flow ops

def conv_model(features, target):
    target = tf.one_hot(target, 10, 1.0, 0.0)
    features = tf.expand_dims(features, 3)
    features = tf.reduce_max(layers.conv2d(features, 12, [3, 3]), [1, 2])
    features = tf.reshape(features, [-1, 12])
    prediction, loss = learn.models.logistic_regression(features, target)
    train_op = layers.optimize_loss(loss,
        tf.contrib.framework.get_global_step(), optimizer='SGD',
        learning_rate=0.01)
    return tf.argmax(prediction, dimension=1), loss, train_op

# Create a classifier, train and predict.
classifier = learn.Estimator(model_fn=conv_model)
classifier.fit(X_train, y_train, steps=1000, batch_size=128)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: %f' % score)
