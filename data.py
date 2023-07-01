from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import cv2
import pandas as pd
import numpy as np

class TrainTestSplit():
    def __init__(self,test_size,seed=2023):
        self.root = 'IMAGES/'
        self.test_size = test_size
        self.seed = seed
    
    def get_content(self,path):
        return os.listdir(path)
    
    def path_to_x_y(self,folder,output_list=[]):
        for image in self.get_content(os.path.join(self.root,folder)):
            to_append = (os.path.join(self.root,folder,image),folder)
            output_list.append(to_append)
        return output_list
    
    def convert_list(self,list):
        path_images, classes = zip(*list)
        return path_images, classes

    def split_train_test(self):
        output_list=[]
        for specie in self.get_content(self.root):
            output_list = self.path_to_x_y(specie,output_list)
        path_images, classes = self.convert_list(output_list)
        X_train, X_test, y_train, y_test = train_test_split(path_images, classes,random_state=self.seed,test_size=self.test_size)
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    t = TrainTestSplit(test_size=0.1)
    X_train, X_test, y_train, y_test = t.split_train_test()
    y = y_train + y_test
    # le = preprocessing.LabelEncoder()
    # le.fit(y)
    # print(le.transform(y_train))
    print(np.array([[0,0,1],[0,0,0]]))
    print(set(y))