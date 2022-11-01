import json
from attr import attrs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import im

img_file_path = '../data/image/2022-04-22-11-00-05004 (98).jpg'
label_file_path = '../data/labeljson/2022-04-22-11-00-05004 (98).json'
road_file_path = '../data/roadjson/2022-04-22-11-00-05004 (98).json'

class ShowImg:
    def __init__(self) -> None:
        pass

    def getCoordinateFromLabel(self, json_path):
        coordinates = []
        with open(json_path, encoding='utf-8') as j:
            contents = json.loads(j.read())
            for instance in contents['instances']:
                for children in instance['children']:
                    for camera in children['cameras']:
                        for frame in camera['frames']:
                            for point in frame['shape']['points']:
                                coordinates.append((int(point['x']), int(point['y'])))
        return coordinates

    def getCoordinateFromRoad(self, road_path):
        coordinates = []
        with open(road_path, encoding='utf-8') as j:
            contents = json.loads(j.read())
            for js_obj in contents['json']:
                for obj in js_obj['objects']:
                    for poly in obj['polygon']:
                        coordinates.append((int(poly['x']), int(poly['y'])))
        return coordinates


    def draw(self, coordinates):
        img = plt.imread(img_file_path)
        for pt in coordinates:
            cv2.circle(img, pt, radius=5, color=(0, 255, 0))
        plt.imshow(img)
        plt.show()
    
    

if __name__ == '__main__':
    testObj = ShowImg()
    label = testObj.getCoordinateFromLabel(label_file_path)
    road = testObj.getCoordinateFromRoad(road_file_path)
    testObj.draw(label)