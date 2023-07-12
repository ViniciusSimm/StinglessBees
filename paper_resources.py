import tensorflow as tf

from data import TrainTestSplit
from utils import PrepareData
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class ConfusionMatrix():
    def __init__(self, model, fold):
        model_path = "./models/{}.h5".format(model)
        self.model = tf.keras.models.load_model(model_path)
        self.fold = fold
        self.classes = PrepareData().convert
        self.order_labels = ['boca_de_sapo', 'tubuna', 'bora', 'jatai', 'mandaguari', 'mirim_droryana', 'mandacaia', 'bugia', 'lambe_olhos', 'japura', 'moca_branca', 'irai', 'mirim_preguica']

    def apply_model(self):
        X_train_paths, X_test_paths, y_train_string, y_test_string = TrainTestSplit(test_size=0.000001).split_train_test()
        X_train = PrepareData().get_images(X_train_paths)
        y_train = PrepareData().encode(y_train_string)

        train_folds_index, test_folds_index = PrepareData().fold_cross(X_train, y_train)

        loss,acc = self.model.evaluate(X_train[test_folds_index[self.fold]],y_train[test_folds_index[self.fold]])

        y_pred = self.model.predict(X_train[test_folds_index[self.fold]])

        y_pred_bin = [list(PrepareData().convert_softmax_to_one_hot(i).astype(int)) for i in y_pred]

        y_pred_string = [PrepareData().decode(i) for i in y_pred_bin]

        answers = [PrepareData().decode(i) for i in y_train[test_folds_index[self.fold]]]

        print(Counter(answers))

        labels_to_matrix = PrepareData().obter_strings_unicas(answers)

        print(labels_to_matrix)

        cm = sklearn.metrics.confusion_matrix(answers,y_pred_string,labels=self.order_labels)

        print(cm)

        return cm,loss,acc
    
    def plot_config(self,sum_cm):
        disp = sklearn.metrics.ConfusionMatrixDisplay(sum_cm,display_labels=['boca_de_sapo', 'tubuna', 'bora', 'jatai', 'mandaguari', 'mirim_droryana', 'mandacaia', 'bugia', 'lambe_olhos', 'japura', 'moca_branca', 'irai', 'mirim_preguica'])
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax,cmap=plt.cm.Blues)
        plt.xticks(rotation=45)
        plt.show()


if __name__ == '__main__':

    sum_cm = np.zeros((13, 13), dtype=int)
    sum_loss = 0
    sum_acc = 0
    for iteration in range(5):
        MODEL = 'DENSENET121_freeze_{}'.format(iteration)
        CM = ConfusionMatrix(model=MODEL,fold=iteration)
        cm,loss,acc = CM.apply_model()
        sum_cm = sum_cm + cm
        sum_loss = sum_loss + loss
        sum_acc = sum_acc + acc
    print(sum_cm)
    print(sum_loss/5)
    print(sum_acc/5)
    CM.plot_config(sum_cm)
    # disp = sklearn.metrics.ConfusionMatrixDisplay(sum_cm,display_labels=['boca_de_sapo', 'tubuna', 'bora', 'jatai', 'mandaguari', 'mirim_droryana', 'mandacaia', 'bugia', 'lambe_olhos', 'japura', 'moca_branca', 'irai', 'mirim_preguica'])
    # disp.plot()
    # plt.show()
