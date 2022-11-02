from cmath import isclose
from turtle import color
from operator import itemgetter
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import json

class TransToTusimple:
    def __init__(self) -> None:
        self.color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
        self.h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
             370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
             580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
    
    def drawLines(self, img_file_path, coordinates):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, coordinate in enumerate(coordinates):
            cv2.polylines(img, np.int32([coordinate]), isClosed=False, color=self.color[idx], thickness=5)
        plt.imshow(img)
        plt.show()

    def drawCircle(self, img_file_path, coordinates):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, line in enumerate(coordinates):
            for pt in line:
                cv2.circle(img, pt, radius=5, color=self.color[idx])
        plt.imshow(img)
        plt.show()
    
    def drawTusimple(self, img_file_path, coordinates):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, coordinate in enumerate(coordinates):
            for pt in zip(coordinate, self.h_samples):
                if pt[0] > 0:
                    cv2.circle(img, pt, radius=5, color=self.color[idx])
        plt.imshow(img)
        plt.show()
    
    def getCoordinateFromLane(self, json_file_path):
        coordinates = []
        with open(json_file_path, encoding='utf-8') as j:
            contents = json.loads(j.read())
            for instance in contents['instances']:
                instanc_list = []
                for children in instance['children']:
                    for camera in children['cameras']:
                        for frame in camera['frames']:
                            for point in frame['shape']['points']:
                                instanc_list.append((int(point['x']*2/3), int(point['y']*2/3)))
                coordinates.append(instanc_list)
        return coordinates
    
    def getAllFiles(self, json_path):
        file_list = os.listdir(json_path)
        return file_list

    def getLinesData(self, coordinates):
        lanes = []
        for line in coordinates:
            if max(line, key=itemgetter(0))[0] - min(line, key=itemgetter(0))[0] > 800:
                continue
            intersection_x = []
            start_idx = len(line)-1
            if start_idx < 1:
                continue
            x1, y1 = line[start_idx]
            x2, y2 = line[start_idx-1]
            for h in self.h_samples:
                if h < min(y1, y2):
                    intersection_x.append(-2)
                    continue
                while (h > max(y1, y2) or y1 == y2) and start_idx - 1 > 0:
                    start_idx -= 1
                    x1, y1 = line[start_idx]
                    x2, y2 = line[start_idx-1]
                if h > max(y1, y2) or y1 == y2:
                    intersection_x.append(-2)
                    continue
                sign = 1
                a = y2 - y1
                if a < 0:
                    sign = -1
                    a = sign * a
                b = sign * (x1 - x2)
                c = sign * (y1 * x2 - x1 * y2)
                x = -(b * h + c) / a
                intersection_x.append(round(x))
            lanes.append(intersection_x)
        return lanes

    def createJson(self, label_path):
        if not os.access(label_path, os.F_OK):
            file = open(label_path, 'a')
            file.close()
            print("create label.json success")
        else:
            print("label.json exist")
    
    def buildJSON(self, json_path, img_path, label_path):
        self.createJson(label_path)
        files_list = self.getAllFiles(json_path)
        for file_path in files_list:
            image_name = file_path.strip('.json')
            image_path = img_path + image_name + '.jpg'
            coordinates = self.getCoordinateFromLane(json_path+file_path)
            lines = self.getLinesData(coordinates=coordinates)
            if not lines:
                continue
            info = {'lanes':lines, 'h_samples':self.h_samples, 'raw_file':image_path}
            fr = open(label_path, 'a')
            model = json.dumps(info)
            fr.write(model)
            fr.write('\r')
            fr.close()
            print(file_path + " " + "handle succeed")
            # self.drawCircle(img_path, coordinates)
            # self.drawTusimple(image_path, lines)

            


if __name__ == '__main__':
    img_path = '../roadData/images/'
    json_path = '../roadData/laneJSON/'
    label_path = '../roadData/self_label.json'
    model = TransToTusimple()
    model.buildJSON(json_path=json_path, img_path=img_path, label_path=label_path)

                
    