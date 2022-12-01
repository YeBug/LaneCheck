import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import im
from operator import itemgetter

# 预测结果
output_path = '../roadData/tusimple_predictions.json'
# gt数据
test_gt_path = '../roadData/test_label_complex.json'

class ShowImg:
    def __init__(self) -> None:
        self.color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

        self.h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
             370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
             580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
    
    def drawLines(self, img_file_path, pre_line, pre_color, gt_line, gt_color):
        img = plt.imread(img_file_path)
        img = cv2.resize(img, (0, 0), fx=2/3, fy=2/3)
        for line in pre_line:
            cv2.polylines(img, np.int32([list(tups for tups in zip(line, self.h_samples) if tups[0] > 0 )]), isClosed=False, color=pre_color, thickness=3) 
        for line in gt_line:
            cv2.polylines(img, np.int32([list(tups for tups in zip(line, self.h_samples) if tups[0] > 0 )]), isClosed=False, color=gt_color, thickness=3) 
        plt.imshow(img)
        plt.show()
        plt.pause(1)
        plt.close()

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
    
    # 预测集和测试集需对应，simple对simple， complex对complex
    def drawGTcompTest(self, output_path, test_gt_path):
        with open(output_path, encoding='utf-8') as pre_jsons, open(test_gt_path, encoding='utf-8') as gt_jsons:
            for line in pre_jsons:
                instance = json.loads(line)
                # img_path = tusimple_path + instance['raw_file']
                img_path = instance['raw_file']
                lanes = instance['lanes']
                gt_instance = json.loads(gt_jsons.readline())
                gt_lanes = gt_instance['lanes']
                self.drawLines(img_path, lanes, self.color[0], gt_lanes, self.color[1])



if __name__ == '__main__':
    testObj = ShowImg()
    plt.ion()
    label = testObj.drawGTcompTest(output_path, test_gt_path)