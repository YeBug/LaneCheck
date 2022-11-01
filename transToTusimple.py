from cmath import isclose
import os
from turtle import color
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

class TransToTusimple:
    def __init__(self) -> None:
        self.img_file_path = '../data/image/2022-04-22-11-00-05004 (98).jpg'
        self.label_file_path = '../data/labeljson/2022-04-22-11-00-05004 (98).json'
        self.color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
        pass
    
    def draw(self, coordinates):
        img = plt.imread(self.img_file_path)
        for idx, coordinate in enumerate(coordinates):
            cv2.polylines(img, np.int32([coordinate]), isClosed=False, color=self.color[idx], thickness=5)
        plt.imshow(img)
        plt.show()
    
    def getCoordinateFromLane(self):
        coordinates = []
        with open(self.label_file_path, encoding='utf-8') as j:
            contents = json.loads(j.read())
            for instance in contents['instances']:
                instanc_list = []
                for children in instance['children']:
                    for camera in children['cameras']:
                        for frame in camera['frames']:
                            for point in frame['shape']['points']:
                                instanc_list.append((point['x'], point['y']))
                coordinates.append(instanc_list)
        self.draw(coordinates)

if __name__ == '__main__':
    model = TransToTusimple()
    model.getCoordinateFromLane()

                
    