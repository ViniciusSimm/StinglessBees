import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

class PrepareData():
    def get_img(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))
        img = img.astype('float32')
        return img
    
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
        convert = {'irai':[1,0,0,0,0,0,0,0,0,0,0,0,0],
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
        output = np.array([convert[i] for i in input])
        return output

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
    history_table = pd.read_csv('history/history.csv')
    history_class = GetHistory(history_table)
    acc = history_class.accuracy_vs_val_accuracy()
    loss = history_class.loss_vs_val_loss()