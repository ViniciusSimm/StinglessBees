import tensorflow as tf

from data import TrainTestSplit
from utils import PrepareData
from architecture import VGG16_MODEL
import sklearn
import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# SETUP
#===============================================================================

MODEL = 'tf_model_v2'

#===============================================================================
# LOAD DATA
#===============================================================================

X_train_paths, X_test_paths, y_train_string, y_test_string = TrainTestSplit(test_size=0.1).split_train_test()
X_test = PrepareData().get_images(X_test_paths)
y_test = PrepareData().encode(y_test_string)

classes = PrepareData().convert

model_path = "./models/{}.h5".format(MODEL)
model = tf.keras.models.load_model(model_path)

loss,acc = model.evaluate(X_test,y_test)

print(loss,acc)

y_pred = model.predict(X_test)
# print(y_pred)
y_pred_bin = [list(PrepareData().convert_softmax_to_one_hot(i).astype(int)) for i in y_pred]
# print(y_pred_bin)
y_pred_string = [PrepareData().decode(i) for i in y_pred_bin]
print(y_pred_string)

cm = sklearn.metrics.confusion_matrix(y_test_string,y_pred_string)
disp = sklearn.metrics.ConfusionMatrixDisplay(cm)

disp.plot()
plt.show()

