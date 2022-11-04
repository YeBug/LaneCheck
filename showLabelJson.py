import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import im
from operator import itemgetter

img_file_path = '../roadData/images/2022-04-23-17-36-58005(92).jpg'
label_file_path = '../roadData/laneJSON/2022-04-23-17-36-58005(92).json'

class ShowImg:
    def __init__(self) -> None:
        self.color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

        self.h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
             370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
             580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]

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

    def drawCircle(self, coordinates):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, line in enumerate(coordinates):
            for pt in line:
                cv2.circle(img, pt, radius=5, color=self.color[idx])
        plt.imshow(img)
        plt.show()
        
    def drawLines(self, coordinates):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, coordinate in enumerate(coordinates):
            cv2.polylines(img, np.int32([coordinate]), isClosed=False, color=self.color[idx], thickness=1)
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
    
    def drawTusimple(self, coordinates):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, coordinate in enumerate(coordinates):
            for pt in zip(coordinate, self.h_samples):
                if pt[0] > 0:
                    cv2.circle(img, pt, radius=5, color=self.color[idx])
        plt.imshow(img)
        plt.show()
    
    def getLinesData(self, coordinates):
        lanes = []
        for line in coordinates:
            if max(line, key=itemgetter(0))[0] - min(line, key=itemgetter(0))[0] > 800 \
                or max(line, key=itemgetter(1))[1] - min(line, key=itemgetter(1))[1] < 60:
                continue
            intersection_x = []
            start_idx = len(line)-1
            if start_idx < 1:
                continue
            x1, y1 = line[start_idx]
            x2, y2 = line[start_idx-1]
            positive_points = 0
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
                positive_points += 1
            if positive_points > 1:
                lanes.append(intersection_x)
        return lanes

if __name__ == '__main__':
    testObj = ShowImg()
    label = testObj.getCoordinateFromLane(label_file_path)
    testObj.drawTusimple(testObj.getLinesData(label))