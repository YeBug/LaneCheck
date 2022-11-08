import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import im
from operator import itemgetter

output_path = '../RemoteProj/LaneDet-ATT/tusimple_predictions.json'

class ShowImg:
    def __init__(self) -> None:
        self.color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

        self.h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
             370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
             580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]

    def drawOutputImg(self):
        for line in open(output_path, encoding='utf-8'):
            instance = json.loads(line)
            img_path = instance['raw_file']
            lanes = instance['lanes']
            self.drawTusimple(lanes, img_path)

    def drawTusimple(self, coordinates, img_path):
        img = plt.imread(img_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for idx, coordinate in enumerate(coordinates):
            for pt in zip(coordinate, self.h_samples):
                if pt[0] > 0:
                    cv2.circle(img, pt, radius=5, color=self.color[idx])
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    testObj = ShowImg()
    label = testObj.drawOutputImg()