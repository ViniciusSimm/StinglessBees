import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from keras.utils import to_categorical

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

if __name__ == '__main__':
    p = PrepareData()
    print(p.encode(['lambe_olhos','boca_de_sapo']))