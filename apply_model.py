import tensorflow as tf

from data import TrainTestSplit
from utils import PrepareData
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#===============================================================================
# SETUP
#===============================================================================

iteration = 3

MODEL = 'DENSENET121_freeze_{}'.format(iteration)

#===============================================================================
# LOAD DATA
#===============================================================================

X_train_paths, X_test_paths, y_train_string, y_test_string = TrainTestSplit(test_size=0.000001).split_train_test()
X_train = PrepareData().get_images(X_train_paths)
y_train = PrepareData().encode(y_train_string)

train_folds_index, test_folds_index = PrepareData().fold_cross(X_train, y_train)

classes = PrepareData().convert

model_path = "./models/{}.h5".format(MODEL)
model = tf.keras.models.load_model(model_path)

loss,acc = model.evaluate(X_train[test_folds_index[iteration]],y_train[test_folds_index[iteration]])

print(loss,acc)

y_pred = model.predict(X_train[test_folds_index[iteration]])
# print(y_pred)
y_pred_bin = [list(PrepareData().convert_softmax_to_one_hot(i).astype(int)) for i in y_pred]
# print(y_pred_bin)
y_pred_string = [PrepareData().decode(i) for i in y_pred_bin]
# print(y_pred_string)

answers = [PrepareData().decode(i) for i in y_train[test_folds_index[iteration]]]

print(Counter(answers))

labels_to_matrix = PrepareData().obter_strings_unicas(answers)

print(labels_to_matrix)

cm = sklearn.metrics.confusion_matrix(answers,y_pred_string,labels=['boca_de_sapo', 'tubuna', 'bora', 'jatai', 'mandaguari', 'mirim_droryana', 'mandacaia', 'bugia', 'lambe_olhos', 'japura', 'moca_branca', 'irai', 'mirim_preguica'])

print(type(cm))

disp = sklearn.metrics.ConfusionMatrixDisplay(cm,display_labels=['boca_de_sapo', 'tubuna', 'bora', 'jatai', 'mandaguari', 'mirim_droryana', 'mandacaia', 'bugia', 'lambe_olhos', 'japura', 'moca_branca', 'irai', 'mirim_preguica'])

disp.plot()
plt.show()

