
import json
import os
 
 
# 计算线段与行直线的交点
def getLinesData(x1, y1, x2, y2):
    intersection_x = []
    for h in h_samples:
        if h < y1 and h < y2:
            intersection_x.append(-2)
        else:
            sign = 1
            a = y2 - y1
            if a < 0:
                sign = -1
                a = sign * a
            b = sign * (x1 - x2)
            c = sign * (y1 * x2 - x1 * y2)
            x = -(b * h + c) / a
            intersection_x.append(int(x))
    return intersection_x
 
 
# 得到文件夹下所有label.json的文件名列表
def getLabelFilesNameList():
    # 遍历文件夹内所有json文件,找出所有标注数据文件
    file_list = os.listdir(clip_json_path)
    json_name = 'json'
    json_list = []
    for file_name in file_list:
        if json_name in file_name and file_name != 'test_label.json':
            json_list.append(file_name)
    return json_list
 
 
# 计算车道线与h_lines的交点,并存储到json文件中
def saveJson(json_list):
    for file_name in json_list:
        file_path = clip_json_path + file_name
        image_name = file_name.strip('.json')
        # 读取直线两个端点
        file = open(file_path, 'r', encoding='utf-8')
        data = json.load(file)
        x1 = int(data['shapes'][0]['points'][0][0])
        y1 = int(data['shapes'][0]['points'][0][1])
        x2 = int(data['shapes'][0]['points'][1][0])
        y2 = int(data['shapes'][0]['points'][1][1])
        intersection_x = getLinesData(x1, y1, x2, y2)
        file.close()
        # 构建test_label.json数据结构,写入数据
        lanes = [intersection_x]
        row_file = 'clips/144/' + image_name + '.jpg'
        info1 = {"lanes": lanes, "h_sample": h_samples, "raw_file": row_file}
        fr = open(test_label_path, 'a')
        model = json.dumps(info1)
        fr.write(model)
        fr.write('\r\n')
        fr.close()
        print(file_name + " " + "handle succeed")
 
 
# 创建test_label.json文件
def createJson():
    if not os.access(test_label_path, os.F_OK):
        file = open(test_label_path, 'a')
        file.close()
        print("create test_label.json success")
    else:
        print("test_label.json exist")
 
 
# 存放结果文件夹路径
folder_path = '/home/hw/project/LaneDetection/datasets/TuSimple_self/'
# label文件存储路径
clip_json_path = folder_path + 'clip_jsons/144/'
# 结果文件名
test_label = 'test_label.json'
# 结果文件路径
test_label_path = folder_path + test_label
 
# 图片行位置
h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
             370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
             580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
 
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('fuck day')
    createJson()
    label_json_list = getLabelFilesNameList()
    saveJson(label_json_list)
