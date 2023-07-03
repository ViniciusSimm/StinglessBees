import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class PrepareData():
    def __init__(self):
        self.convert = {'irai':[1,0,0,0,0,0,0,0,0,0,0,0,0],
                    'mirim_droryana':[0,1,0,0,0,0,0,0,0,0,0,0,0],
                    'mirim_preguica':[0,0,1,0,0,0,0,0,0,0,0,0,0],
                    'jatai':[0,0,0,1,0,0,0,0,0,0,0,0,0],
                    'bugia':[0,0,0,0,1,0,0,0,0,0,0,0,0],
                    'japura':[0,0,0,0,0,1,0,0,0,0,0,0,0],
                    'mandaguari':[0,0,0,0,0,0,1,0,0,0,0,0,0],
                    'moca_branca':[0,0,0,0,0,0,0,1,0,0,0,0,0],
                    'bora':[0,0,0,0,0,0,0,0,1,0,0,0,0],
                    'tubuna':[0,0,0,0,0,0,0,0,0,1,0,0,0],
                    'boca_de_sapo':[0,0,0,0,0,0,0,0,0,0,1,0,0],
                    'lambe_olhos':[0,0,0,0,0,0,0,0,0,0,0,1,0],
                    'mandacaia':[0,0,0,0,0,0,0,0,0,0,0,0,1]}

    def get_img(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))
        img = img.astype('float32')
        return img
    
    def convert_softmax_to_one_hot(self,softmax_array):
        max_position = np.argmax(softmax_array)
        one_hot_array = np.zeros_like(softmax_array)
        one_hot_array[max_position] = 1
        return one_hot_array

    def get_images(self,list_of_paths):
        all_images = []
        for path in list_of_paths:
            all_images.append(self.get_img(path))
        arrays = np.array(all_images)
        return arrays
    
    def create_label_encoder(self,labels):
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder

    # def encode(self,encoder,list):
    #     labels_encoded = encoder.transform(list)
    #     labels_categorical = to_categorical(labels_encoded)
    #     return np.array(labels_categorical)

    def encode(self,input):
        output = np.array([self.convert[i] for i in input])
        return output
    
    # def decode(self,input):
    #     vec = [list(PrepareData().convert_softmax_to_one_hot(i).astype(int)) for i in input]
    #     return [self.convert.keys()[self.convert.values().index(i)] for i in vec]

    def decode(self,input):
        reverse_convert = {tuple(value): key for key, value in self.convert.items()}
        name = reverse_convert.get(tuple(input))
        return name

    def fold_cross(self, X, y):
        skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        skf.get_n_splits(X, y)

        train_folds_index = []
        test_folds_index = []

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            train_folds_index.append(train_index)
            test_folds_index.append(test_index)
        return np.array(train_folds_index), np.array(test_folds_index)

class GetHistory():
    def __init__(self, df):
        self.df = df
    
    def accuracy_vs_val_accuracy(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.df.index, self.df['accuracy'], label='Train')
        ax.plot(self.df.index, self.df['val_accuracy'], label='Validation')
        ax.set_title('Train and Validation Accuracy',
                     fontdict={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_xlabel('Epochs', fontdict={'fontsize': 12})
        ax.set_ylabel('Accuracy', fontdict={'fontsize': 12})
        colors = ['#1a3cad', '#ff0e26']
        for i, line in enumerate(ax.get_lines()):
            line.set_color(colors[i])
            line.set_linewidth(2.5)
        ax.legend()
        plt.show()
        return fig
        
    def loss_vs_val_loss(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.df.index, self.df['loss'], label='Train')
        ax.plot(self.df.index, self.df['val_loss'], label='Validation')
        ax.set_title('Train and Validation Loss',
                     fontdict={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_xlabel('Epochs', fontdict={'fontsize': 12})
        ax.set_ylabel('Loss', fontdict={'fontsize': 12})
        colors = ['#1a3cad', '#ff0e26']
        for i, line in enumerate(ax.get_lines()):
            line.set_color(colors[i])
            line.set_linewidth(2.5)
        ax.legend()
        plt.show()
        return fig



if __name__ == '__main__':
    history_table = pd.read_csv('history/densenet_v1.csv')
    history_class = GetHistory(history_table)
    acc = history_class.accuracy_vs_val_accuracy()
    loss = history_class.loss_vs_val_loss()
    # data = np.array([0,0.1,0.0111,0.9])
    # print(PrepareData().convert_softmax_to_one_hot(data))
    # print(PrepareData().decode([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))