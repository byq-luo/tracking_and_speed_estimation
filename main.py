#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
import math
from timeit import time
import warnings
from collections import OrderedDict
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YoLo4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "input_data/CH6_preset1_sample.avi")
ap.add_argument("-c", "--class",help="name of class", default = "car")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
ctime = [deque(maxlen=30) for _ in range(9999)]

warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# Define the channel number
ch = 4
# convert pixel to global
if ch == 8:
    def convPtoR_1(x1,y1,x2,y2) :
        w1 = (-3.97334700e-04 *x1) + (-7.81802057e-03 *y1) +1
        w2 = (-3.97334700e-04 *x2) + (-7.81802057e-03 *y2) +1

        Rx1 = ((-1.48158895e-02*x1)+(-2.91535103e-01*y1) +3.72901223e+01)/w1
        Ry1 = ((-5.05122054e-02 *x1) +(-9.93890669e-01 * y1) + 1.27126128e+02)/w1
        Rx2 = ((-1.48158895e-02*x2)+(-2.91535103e-01*y2) +3.72901223e+01)/w2
        Ry2 = ((-5.05122054e-02 *x2) +(-9.93890669e-01 * y2) + 1.27126128e+02)/w2


        x_distance = (abs(Rx1-Rx2) * 100000 *1.1 ) #보정
        y_distance = (abs(Ry1-Ry2) * 100000 *0.9)
        R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
        return R_dist

    def convPtoR_2(x1,y1,x2,y2) :
        w1 = (-1.65876404e-04 *x1) + (-2.02321289e-03 *y1) +1
        w2 = (-1.65876404e-04 *x2) + (-2.02321289e-03 *y2) +1

        Rx1 = ((-6.18533090e-03*x1)+(-7.54458945e-02*y1) +3.72904094e+01)/w1
        Ry1 = ((-2.10876492e-02 *x1) +(-2.57207363e-01 * y1) + 1.27126007e+02)/w1
        Rx2 = ((-6.18533090e-03*x2)+(-7.54458945e-02*y2) +3.72904094e+01)/w2
        Ry2 = ((-2.10876492e-02 *x2) +(-2.57207363e-01 * y2) + 1.27126007e+02)/w2
        
        x_distance = (abs(Rx1-Rx2) * 100000 *1.1 ) #보정
        y_distance = (abs(Ry1-Ry2) * 100000 *0.9)
        R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
        return R_dist
    
elif ch == 4:
    # convert pixel to global
    def convPtoR_1(x1,y1,x2,y2) :
        w1 = (1.84179904e-04 *x1) + (-2.79480887e-03 *y1) +1
        w2 = (1.84179904e-04 *x2) + (-2.79480887e-03 *y2) +1

        Rx1 = ((6.86863713e-03*x1)+(-1.04222832e-01*y1) +3.72913945e+01)/w1
        Ry1 = ((2.34118019e-02 *x1) +(-3.55257579e-01 * y1) + 1.27112638e+02)/w1
        Rx2 = ((6.86863713e-03*x2)+(-1.04222832e-01*y2) +3.72913945e+01)/w2
        Ry2 = ((2.34118019e-02 *x2) +(-3.55257579e-01 * y2) + 1.27112638e+02)/w2


        x_distance = (abs(Rx1-Rx2) * 100000 *1.1 ) #보정
        y_distance = (abs(Ry1-Ry2) * 100000 *0.9)
        R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
        return R_dist

    def convPtoR_2(x1,y1,x2,y2) :
        w1 = (1.35319198e-04 *x1) + (-3.63551791e-03 *y1) +1
        w2 = (1.35319198e-04 *x2) + (-3.63551791e-03 *y2) +1

        Rx1 = ((5.04663096e-03*x1)+(-1.35574220e-01*y1) +3.72911979e+01)/w1
        Ry1 = ((1.72008339e-02 *x1) +(-4.62122992e-01 * y1) + 1.27110210e+02)/w1
        Rx2 = ((5.04663096e-03*x2)+(-1.35574220e-01*y2) +3.72911979e+01)/w2
        Ry2 = ((1.72008339e-02 *x2) +(-4.62122992e-01 * y2) + 1.27110210e+02)/w2
        
        x_distance = (abs(Rx1-Rx2) * 100000 *1.1 ) #보정
        y_distance = (abs(Ry1-Ry2) * 100000 *0.9)
        R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
        return R_dist
    
lk_params = dict( winSize  = (13, 13),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

################# parameters ######################
preset = 1
p1_flag = False
p2_flag = False

vline1 = []
mid_line = []
vline2 = []

lane_1 = []
lane_2 = []
lane_3 = []
lane_4 = []
lane_5 = []
lane_6 = []

mask = []
global_point = []


cnt_lane_1 = 0
cnt_lane_2 = 0
cnt_lane_3 = 0
cnt_lane_4 = 0
cnt_lane_5 = 0
cnt_lane_6 = 0

speed_lane_1 = []
speed_lane_2 = []
speed_lane_3 = []
speed_lane_4 = []
speed_lane_5 = []
speed_lane_6 = []
##################################################

# visualization part
if ch == 8:
    # preset 1
    p1_vline1 = [(359,338), (794,314)]
    p1_mid_line = [(324,485), (1007,445)]
    p1_vline2 = [(86,1447), (1076,1364)]

    p1_mask = np.array([[0, 0], [0, 1008], [356, 231], [747, 220], [1075, 344], [1080, 0]], np.int32)

    p1_lane_1 = np.array([[685, 255], [780, 309], [936, 402], [1079, 489], [1078, 572], [929, 470], [851, 414], [758, 346], [696, 302], [637, 258]], np.int32)
    p1_lane_2 = np.array([[638, 260], [692, 302], [755, 350], [848, 417], [1022, 546], [1078, 584], [1078, 761], [874, 552], [738, 416], [671, 347], [628, 305], [585, 261]], np.int32)
    p1_lane_3 = np.array([[529, 262], [557, 307], [587, 351], [636, 422], [729, 560], [943, 896], [1079, 1095], [1078, 771], [869, 552], [734, 419], [666, 350], [622, 305], [580, 261]], np.int32)
    p1_lane_4 = np.array([[477, 265], [490, 308], [531, 431], [574, 560], [690, 901], [846, 1373], [1028, 1911], [1075, 1912], [1074, 1122], [934, 900], [720, 562], [632, 426], [554, 309], [527, 264]], np.int32)
    p1_lane_5 = np.array([[425, 265], [425, 338], [431, 577], [440, 926], [457, 1408], [476, 1916], [1006, 1916], [834, 1374], [681, 897], [570, 560], [504, 353], [475, 267]], np.int32)
    p1_lane_6 = np.array([[377, 267], [357, 343], [282, 635], [185, 1017], [77, 1478], [0, 1810], [1, 1917], [448, 1916], [437, 1215], [429, 663], [421, 338], [420, 262]], np.int32)

    p1_global_point = []

    # read global point file
    f = open('input_data/CCTV8_PRESET1.txt', 'r')
 
    # preset 2
    p2_vline1 = [(232,843), (967,806)]
    p2_mid_line = [(188,1015), (1076,973)]
    p2_vline2 = [(41,1502), (1076,1472)]

    p2_mask = np.array([[0, 0], [5, 1399], [264, 731], [924, 731], [1080, 800], [1080, 0]], np.int32)

    p2_lane_1 = np.array([[897, 764], [1079, 869], [1077, 968], [932, 869], [865, 824], [788, 769]], np.int32)
    p2_lane_2 = np.array([[673, 775], [768, 857], [852, 940], [978, 1060], [1078, 1156], [1075, 977], [926, 871], [838, 813], [779, 771]], np.int32)
    p2_lane_3 = np.array([[666, 775], [775, 876], [868, 964], [930, 1023], [1008, 1099], [1077, 1169], [1079, 1526], [909, 1277], [760, 1068], [680, 946], [602, 835], [563, 775]], np.int32)
    p2_lane_4 = np.array([[556, 777], [627, 884], [726, 1031], [847, 1211], [949, 1355], [1078, 1549], [1077, 1917], [839, 1912], [692, 1472], [591, 1168], [515, 954], [475, 839], [457, 781]], np.int32)
    p2_lane_5 = np.array([[454, 782], [501, 924], [565, 1117], [643, 1361], [716, 1574], [822, 1917], [353, 1917], [354, 1231], [351, 965], [349, 782]], np.int32)
    p2_lane_6 = np.array([[342, 786], [344, 977], [343, 1269], [340, 1563], [331, 1915], [4, 1916], [0, 1638], [134, 1199], [204, 961], [252, 787]], np.int32)

    p2_global_point = []

    # read global point file
    f = open('input_data/CCTV8_PRESET2.txt', 'r')
    lines = f.readlines()

elif ch == 4:
    p1_mask = np.array([[0,0], [1, 804], [512, 516], [1077, 528], [1080, 0]], np.int32)
    p1_lane_1 = np.array([[1030, 565], [1004, 586], [988, 600], [979, 615], [969, 625], [964, 640], [955, 655], [950, 675], [943, 692], [938, 730], [943, 782], [960, 850], [990, 935], [1020, 993], [1059, 1058], [1078, 1093], [1077, 720], [1072, 697], [1073, 676], [1074, 641], [1077, 563]], np.int32)
    p1_lane_2 = np.array([[1030, 565], [1004, 586], [988, 600], [979, 615], [969, 625], [964, 640], [955, 655], [950, 675], [943, 692], [938, 730], [943, 782], [960, 850], [990, 935], [1020, 993], [1059, 1058], [1078, 1093], [1077, 720], [1072, 697], [1073, 676], [1074, 641], [1077, 563]], np.int32)
    p1_lane_3 = np.array([[1022, 565], [1002, 583], [984, 598], [965, 625], [948, 655], [938, 690], [933, 729], [936, 784], [951, 848], [982, 934], [1011, 993], [1078, 1113], [1077, 1731], [982, 1515], [914, 1348], [829, 1135], [786, 979], [773, 885], [773, 811], [784, 752], [800, 702], [820, 669], [847, 636], [872, 608], [904, 582], [936, 560]], np.int32)
    p1_lane_4 = np.array([[931, 557], [903, 582], [870, 606], [844, 632], [816, 666], [798, 702], [781, 750], [768, 807], [765, 883], [779, 982], [796, 1049], [823, 1137], [848, 1223], [896, 1349], [964, 1517], [1074, 1772], [1074, 1917], [600, 1916], [571, 1744], [558, 1498], [545, 1329], [539, 1119], [548, 973], [567, 874], [600, 799], [629, 746], [660, 698], [694, 664], [745, 615], [786, 590], [844, 554]], np.int32)
    p1_lane_5 = np.array([[837, 555], [803, 577], [761, 603], [727, 630], [693, 660], [656, 700], [623, 742], [592, 799], [563, 872], [538, 975], [529, 1119], [531, 1327], [555, 1752], [564, 1915], [25, 1916], [142, 1451], [240, 1127], [272, 1051], [324, 939], [385, 839], [451, 758], [519, 695], [560, 662], [613, 626], [672, 594], [723, 563], [760, 549]], np.int32)
    p1_lane_6 = np.array([[753, 548], [697, 571], [638, 604], [580, 638], [539, 670], [471, 725], [410, 796], [351, 872], [298, 961], [242, 1081], [192, 1224], [123, 1446], [57, 1695], [1, 1891], [1, 1103], [108, 966], [206, 853], [274, 789], [334, 738], [419, 682], [493, 635], [567, 595], [632, 570], [685, 543]], np.int32)
    
    p1_global_point = []

    # read global point file
    f = open('input_data/CCTV8_PRESET1.txt', 'r')
    lines = f.readlines()

    #작은 노란점들? 마커 좌표 저장
    for line in lines:
        x = line.split(',')[0]
        y = line.split(',')[1]
        y = y.split('\n')[0]

        x = int(x)
        y = int(y)

        if x >= 0 and y >= 0:
            p1_global_point.append((x,y))
    f.close()

    p2_mask = np.array([[0, 0], [2, 1188], [181, 847], [291, 632], [329, 528], [731, 525], [1075, 752], [1080, 0]], np.int32)
    p2_lane_1 = np.array([[673, 560], [696, 588], [736, 633], [797, 695], [854, 747], [917, 805], [1015, 895], [1078, 951], [1078, 832], [999, 778], [916, 714], [843, 662], [781, 613], [743, 584], [719, 560]], np.int32)
    p2_lane_2 = np.array([[671, 559], [691, 590], [732, 633], [795, 696], [849, 746], [910, 806], [1010, 898], [1077, 961], [1076, 1168], [953, 1021], [845, 879], [778, 799], [694, 691], [651, 630], [625, 587], [613, 556]], np.int32)
    p2_lane_3 = np.array([[613, 559], [627, 588], [652, 628], [690, 687], [776, 799], [837, 881], [947, 1020], [1077, 1177], [1078, 1631], [873, 1250], [749, 1007], [685, 879], [641, 792], [598, 690], [574, 626], [563, 583], [559, 557]], np.int32)
    p2_lane_4 = np.array([[557, 555], [561, 582], [572, 625], [594, 686], [639, 792], [741, 1006], [861, 1246], [1077, 1652], [1076, 1913], [716, 1916], [677, 1701], [581, 1245], [535, 1006], [516, 869], [503, 795], [496, 678], [495, 621], [495, 583], [502, 552]], np.int32)
    p2_lane_5 = np.array([[499, 554], [492, 581], [490, 620], [491, 680], [499, 795], [528, 1009], [571, 1254], [652, 1700], [687, 1911], [191, 1913], [282, 1255], [324, 1002], [366, 794], [396, 675], [412, 620], [426, 583], [447, 551]], np.int32)
    p2_lane_6 = np.array([[440, 552], [424, 581], [408, 615], [374, 731], [343, 853], [297, 1093], [256, 1340], [204, 1687], [165, 1917], [3, 1917], [2, 1333], [153, 985], [246, 793], [335, 624], [362, 581], [391, 550]], np.int32)
    
    p2_global_point = []

    # read global point file
    f = open('input_data/CCTV8_PRESET2.txt', 'r')
    lines = f.readlines()

# save the global points
for line in lines:
    x = line.split(',')[0]
    y = line.split(',')[1]
    y = y.split('\n')[0]

    x = int(x)
    y = int(y)

    if x >= 0 and y >= 0:
        p2_global_point.append((x,y))
f.close()

# reshape
p1_mask = p1_mask.reshape((-1, 1, 2))
p1_lane_1 = p1_lane_1.reshape((-1, 1, 2))
p1_lane_2 = p1_lane_2.reshape((-1, 1, 2))
p1_lane_3 = p1_lane_3.reshape((-1, 1, 2))
p1_lane_4 = p1_lane_4.reshape((-1, 1, 2))
p1_lane_5 = p1_lane_5.reshape((-1, 1, 2))
p1_lane_6 = p1_lane_6.reshape((-1, 1, 2))

# reshape
p2_mask = p2_mask.reshape((-1, 1, 2))
p2_lane_1 = p2_lane_1.reshape((-1, 1, 2))
p2_lane_2 = p2_lane_2.reshape((-1, 1, 2))
p2_lane_3 = p2_lane_3.reshape((-1, 1, 2))
p2_lane_4 = p2_lane_4.reshape((-1, 1, 2))
p2_lane_5 = p2_lane_5.reshape((-1, 1, 2))
p2_lane_6 = p2_lane_6.reshape((-1, 1, 2))

def check_vline(p1, p2, w):
    x = (p2[0]-p1[0])*(w[1]-p1[1])-(w[0]-p1[0])*(p2[1]-p1[1])
    return x

def main(yolo):
    global p1_flag, p2_flag, vline1, mid_line, vline2, lane_1, lane_2, lane_3, lane_4, lane_5, lane_6, mask, preset
    global cnt_lane_1, cnt_lane_2, cnt_lane_3, cnt_lane_4, cnt_lane_5, cnt_lane_6, global_point
    global speed_lane_1, speed_lane_2, speed_lane_3, speed_lane_4, speed_lane_5, speed_lane_6

    ################# parameters ######################
    track_len = 2
    detect_interval = 4
    of_track = []
    preset = 0
    alpha = 0.3
    mm1, mm2, mm3, mm4, mm5, mm6 = 0, 0, 0, 0, 0, 0
    v1, v2, v3, v4, v5, v6 = 0, 0, 0, 0, 0,0
    ptn1, ptn2, ptn3, ptn4, ptn5, ptn6 = 0, 0, 0, 0, 0,0
    prv1, prv2, prv3, prv4, prv5, prv6 = 0, 0, 0, 0, 0, 0
    ms2kmh = 3.6
    fps = 30
    
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3
    
    counter = []
    ##################################################

    # Deep SORT
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    #video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output/CH4_output.avi', fourcc, video_fps, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    frame_idx = 0
    speed_dict = OrderedDict()

    ret, first_frame = video_capture.read()  
    cal_mask1 = np.zeros_like(first_frame[:, :, 0])
    cal_mask2 = np.zeros_like(first_frame[:, :, 0])

    while True:
        ret, frame = video_capture.read()  
        glob = frame.copy()
        cmask = frame.copy()
        
        # Channel and preset setting
        if ch == 8:
            if 0 <= frame_idx <= 480 or 2415 <= frame_idx <= 4203 or 6140 <=frame_idx<=7925 or 9864 <=frame_idx<=11648 or 13585 <= frame_idx <= 15370 or frame_idx >=17306:
                preset = 1
            elif 559 <= frame_idx <= 2340 or 4275 <= frame_idx<=6064 or 8000 <= frame_idx<=9787 or 11730 <=frame_idx<=13513 or 15450 <=frame_idx<=17237:
                preset = 2
            else:
                preset = 0

            if preset == 1:
                vline1 = p1_vline1
                mid_line = p1_mid_line
                vline2 = p1_vline2

                lane_1 = p1_lane_1
                lane_2 = p1_lane_2
                lane_3 = p1_lane_3
                lane_4 = p1_lane_4
                lane_5 = p1_lane_5
                lane_6 = p1_lane_6

                global_point = p1_global_point

                mask = p1_mask
                
                # Polyline
                cv2.polylines(frame, [lane_1], True, (153,255,255))
                cv2.polylines(frame, [lane_2], True, (255,204,204))
                cv2.polylines(frame, [lane_3], True, (204,255,204))
                cv2.polylines(frame, [lane_4], True, (255,204,255))
                cv2.polylines(frame, [lane_5], True, (153,153,255))
                cv2.polylines(frame, [lane_6], True, (102,255,153))

                frame = cv2.line(frame, vline1[0], vline1[1], (0,0,255), 1)
                frame = cv2.line(frame, mid_line[0], mid_line[1], (0,0,255), 1)
                frame = cv2.line(frame, vline2[0], vline2[1], (0,0,255), 1)

                p1_flag = True

                view_polygon = np.array([[10, 1920], [380, 250], [680, 250], [1080, 480], [1080, 1920]])
                cal_polygon = np.array([[361, 304], [755, 293], [1076, 480], [1077, 1067], [163, 1051]])

                pg1 = np.array([[382, 347], [359, 347], [236, 833], [272, 832]])  # RT, LT, LB, RB
                pg2 = np.array([[460, 347], [434, 346], [456, 833], [505, 833]])  # LB, RT, LT, RB
                pg3 = np.array([[544, 345], [514, 345], [686, 833], [755, 832]])  # LB, LT, RT, RB
                pg4 = np.array([[630, 342], [598, 343], [924, 829], [991, 829]])  # LB, LT, LB, RB
                pg5 = np.array([[725, 343], [696, 345], [996, 650], [1056, 646]])  # RT, LB, LT, RB
                pg6 = np.array([[798, 340], [761, 340], [1037, 535], [1070, 530]])  # RT, LB, LT, RB

                cv2.fillConvexPoly(cal_mask1, cal_polygon, 1)
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=cal_mask1)

                if pg1.size>0:
                    cv2.fillPoly(cmask, [pg1], (120, 0, 120), cv2.LINE_AA)
                if pg2.size>0:
                    cv2.fillPoly(cmask, [pg2], (120, 120, 0), cv2.LINE_AA)
                if pg3.size>0:
                    cv2.fillPoly(cmask, [pg3], (0, 120, 120), cv2.LINE_AA)
                if pg4.size>0:
                    cv2.fillPoly(cmask, [pg4], (80, 0, 255), cv2.LINE_AA)
                if pg5.size>0:
                    cv2.fillPoly(cmask, [pg5], (255, 0, 80), cv2.LINE_AA)
                if pg6.size>0:
                    cv2.fillPoly(cmask, [pg6], (120, 0, 0), cv2.LINE_AA)

            elif preset == 2:
                vline1 = p2_vline1
                mid_line = p2_mid_line
                vline2 = p2_vline2

                lane_1 = p2_lane_1
                lane_2 = p2_lane_2
                lane_3 = p2_lane_3
                lane_4 = p2_lane_4
                lane_5 = p2_lane_5
                lane_6 = p2_lane_6
                global_point = p2_global_point

                mask = p2_mask
                
                # Polyline
                cv2.polylines(frame, [lane_1], True, (153,255,255))
                cv2.polylines(frame, [lane_2], True, (255,204,204))
                cv2.polylines(frame, [lane_3], True, (204,255,204))
                cv2.polylines(frame, [lane_4], True, (255,204,255))
                cv2.polylines(frame, [lane_5], True, (153,153,255))
                cv2.polylines(frame, [lane_6], True, (102,255,153))

                frame = cv2.line(frame, vline1[0], vline1[1], (0,0,255), 1)
                frame = cv2.line(frame, mid_line[0], mid_line[1], (0,0,255), 1)
                frame = cv2.line(frame, vline2[0], vline2[1], (0,0,255), 1)

                p2_flag = True

                view_polygon = np.array([[284, 649], [0, 1629], [1076, 1574], [1079, 888], [676, 634]])
                cal_polygon = np.array([[896, 778], [244, 794], [105, 1271], [1077, 1245], [1077, 879]])

                pg1 = np.array([[276, 846], [234, 847], [134, 1200], [199, 1198]])  # RT, LT, LB, RB
                pg2 = np.array([[418, 844], [375, 844], [384, 1196], [442, 1198]])  # LB, RT, LT, RB
                pg3 = np.array([[553, 843], [508, 844], [637, 1194], [706, 1194]])  # LB, LT, RT, RB
                pg4 = np.array([[686, 841], [637, 843], [886, 1190], [968, 1189]])  # LB, LT, LB, RB
                pg5 = np.array([[817, 837], [773, 841], [1005, 1051], [1060, 1047]])  # RT, LB, LT, RB
                pg6 = np.array([[966, 837], [919, 840], [1043, 929], [1087, 927]])  # RT, LT, LB, RB

                cv2.fillConvexPoly(cal_mask2, cal_polygon, 1)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=cal_mask2)

                if pg1.size > 0:
                    cv2.fillPoly(cmask, [pg1], (120, 0, 120), cv2.LINE_AA)
                if pg2.size > 0:
                    cv2.fillPoly(cmask, [pg2], (120, 120, 0), cv2.LINE_AA)
                if pg3.size > 0:
                    cv2.fillPoly(cmask, [pg3], (0, 120, 120), cv2.LINE_AA)
                if pg4.size > 0:
                    cv2.fillPoly(cmask, [pg4], (80, 0, 255), cv2.LINE_AA)
                if pg5.size > 0:
                    cv2.fillPoly(cmask, [pg5], (255, 0, 80), cv2.LINE_AA)
                if pg6.size > 0:
                    cv2.fillPoly(cmask, [pg6], (120, 0, 0), cv2.LINE_AA)

        elif ch == 4:
            if 0 <= frame_idx <= 1751 or frame_idx >= 3655:
                preset = 1
            elif 1797 <= frame_idx <= 3600:
                preset = 2
            else:
                preset = 0

            if preset == 1:
                lane_1 = p1_lane_1
                lane_2 = p1_lane_2
                lane_3 = p1_lane_3
                lane_4 = p1_lane_4
                lane_5 = p1_lane_5
                lane_6 = p1_lane_6

                global_point = p1_global_point

                mask = p1_mask

                # Polyline
                # cv2.polylines(frame, [lane_1], True, (153,255,255))
                # cv2.polylines(frame, [lane_2], True, (255,204,204))
                # cv2.polylines(frame, [lane_3], True, (204,255,204))
                # cv2.polylines(frame, [lane_4], True, (255,204,255))
                # cv2.polylines(frame, [lane_5], True, (153,153,255))
                # cv2.polylines(frame, [lane_6], True, (102,255,153))

                p1_flag = True

                view_polygon = np.array([[731, 563], [385, 567], [33, 1260], [1077, 1254], [1078, 812]])
                cal_polygon = np.array([[914, 669],[286, 675],  [89, 1083], [1078, 1083], [1078, 772]])

                pg6 = np.array([[346, 686], [313, 686], [163, 992], [244, 996]])  # RT, LT, LB, RB
                pg5 = np.array([[430, 684], [401, 685], [338, 998], [420, 1000]])  # LB, RT, LT, RB
                pg4 = np.array([[534, 685], [506, 685], [547, 999], [631, 999]])  # LB, LT, RT, RB
                pg3 = np.array([[654, 685], [609, 684], [760, 1000], [839, 999]])  # LB, LT, LB, RB
                pg2 = np.array([[770, 685], [723, 684], [979, 999], [1051, 998]])  # RT, LB, LT, RB
                pg1 = np.array([[858, 683], [815, 683], [1031, 860], [1077, 857]])  # RT, LB, LT, RB

                cv2.fillConvexPoly(cal_mask1, cal_polygon, 1)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=cal_mask1)

                if pg1.size > 0:
                    cv2.fillPoly(cmask, [pg1], (120, 0, 120), cv2.LINE_AA)
                if pg2.size > 0:
                    cv2.fillPoly(cmask, [pg2], (120, 120, 0), cv2.LINE_AA)
                if pg3.size > 0:
                    cv2.fillPoly(cmask, [pg3], (0, 120, 120), cv2.LINE_AA)
                if pg4.size > 0:
                    cv2.fillPoly(cmask, [pg4], (80, 0, 255), cv2.LINE_AA)
                if pg5.size > 0:
                    cv2.fillPoly(cmask, [pg5], (255, 0, 80), cv2.LINE_AA)
                if pg6.size > 0:
                    cv2.fillPoly(cmask, [pg6], (120, 0, 0), cv2.LINE_AA)

            elif preset == 2:
                lane_1 = p2_lane_1
                lane_2 = p2_lane_2
                lane_3 = p2_lane_3
                lane_4 = p2_lane_4
                lane_5 = p2_lane_5
                lane_6 = p2_lane_6
                global_point = p2_global_point

                mask = p2_mask

                # Polyline
                # cv2.polylines(frame, [lane_1], True, (153,255,255))
                # cv2.polylines(frame, [lane_2], True, (255,204,204))
                # cv2.polylines(frame, [lane_3], True, (204,255,204))
                # cv2.polylines(frame, [lane_4], True, (255,204,255))
                # cv2.polylines(frame, [lane_5], True, (153,153,255))
                # cv2.polylines(frame, [lane_6], True, (102,255,153))

                p2_flag = True

                view_polygon = np.array([[547, 609], [0, 1109], [1, 1271], [1078, 1278], [1079, 594]])
                cal_polygon = np.array([[529, 611], [8, 1105], [1077, 1110], [1078, 599]])

                pg6 = np.array([[556, 609], [493, 607], [108, 1033], [190, 1030]])  # RT, LT, LB, RB
                pg5 = np.array([[693, 604], [642, 602], [356, 1020], [455, 1020]])  # LB, RT, LT, RB
                pg4 = np.array([[812, 633], [765, 633], [604, 1026], [702, 1026]])  # LB, LT, RT, RB
                pg3 = np.array([[932, 638], [882, 636], [883, 1007], [953, 1001]])  # LB, LT, LB, RB
                pg2 = np.array([[1059, 641], [978, 638], [1028, 941], [1079, 916]])  # RT, LB, LT, RB
                pg1 = np.array([])

                cv2.fillConvexPoly(cal_mask2, cal_polygon, 1)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=cal_mask2)

                if pg1.size > 0:
                    cv2.fillPoly(cmask, [pg1], (120, 0, 120), cv2.LINE_AA)
                if pg2.size > 0:
                    cv2.fillPoly(cmask, [pg2], (120, 120, 0), cv2.LINE_AA)
                if pg3.size > 0:
                    cv2.fillPoly(cmask, [pg3], (0, 120, 120), cv2.LINE_AA)
                if pg4.size > 0:
                    cv2.fillPoly(cmask, [pg4], (80, 0, 255), cv2.LINE_AA)
                if pg5.size > 0:
                    cv2.fillPoly(cmask, [pg5], (255, 0, 80), cv2.LINE_AA)
                if pg6.size > 0:
                    cv2.fillPoly(cmask, [pg6], (120, 0, 0), cv2.LINE_AA)

        if ret != True:
            break
        t1 = time.time()
        
        cnt_lane_1 = cnt_lane_2 = cnt_lane_3 = cnt_lane_4 = cnt_lane_5 = cnt_lane_6 = 0

        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        
        #features is 128-dimension vector for each bounding box
        features = encoder(frame,boxs)        
        
        # score to 1.0 here.
        # score 1.0 means that bbox's class score after yolo is 100%
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)

        # detections is object detection result at current frame
        # it may contain many bbox or no bbox
        detections = [detections[i] for i in indices]

        # calculate lane by lane avg speed
        ## swhan method (optical flow)
        mm1, mm2, mm3, mm4, mm5, mm6 = 0, 0, 0, 0, 0,0

        if len(of_track) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in of_track]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_of_tracks = []
            for tr, (x, y), good_flag in zip(of_track, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_of_tracks.append(tr)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            of_track = new_of_tracks

            
            for idx, tr in enumerate(of_track):
                #print(frame_idx, tr)
                if pg1.size > 0:
                    result_pg1 = cv2.pointPolygonTest(pg1, tr[0],True)
                else:
                    result_pg1 = -999
                if pg2.size > 0:
                    result_pg2 = cv2.pointPolygonTest(pg2, tr[0], True)
                else:
                    result_pg2 = -999
                if pg3.size > 0:
                    result_pg3 = cv2.pointPolygonTest(pg3, tr[0], True)
                else:
                    result_pg3 = -999
                if pg4.size > 0:
                    result_pg4 = cv2.pointPolygonTest(pg4, tr[0], True)
                else:
                    result_pg5 = -999
                if pg5.size > 0:
                    result_pg5 = cv2.pointPolygonTest(pg5, tr[0], True)
                else:
                    result_pg5 = -999
                if pg6.size > 0:
                    result_pg6 = cv2.pointPolygonTest(pg6, tr[0], True)
                else:
                    result_pg6 = -999

                if frame_idx % detect_interval == 0:
                    if result_pg1 > 0:
                        ptn1 += 1
                        if preset == 1:
                            mm1 += convPtoR_1(tr[0][0],tr[0][1],tr[1][0],tr[1][1])   
                        elif preset == 2:
                            mm1 += convPtoR_2(tr[0][0],tr[0][1],tr[1][0],tr[1][1])          
                        mmm1 = mm1/ptn1
                        v1 = mmm1*fps*ms2kmh*6
                    if result_pg2 > 0:
                        ptn2 += 1
                        if preset == 1:
                            mm2 += convPtoR_1(tr[0][0],tr[0][1],tr[1][0],tr[1][1])           
                        elif preset == 2:
                            mm2 += convPtoR_2(tr[0][0],tr[0][1],tr[1][0],tr[1][1])          
                        mmm2 = mm2/ptn2
                        v2 = mmm2*fps*ms2kmh*6
                    if result_pg3 > 0:
                        ptn3 += 1
                        if preset == 1:
                            mm3 += convPtoR_1(tr[0][0],tr[0][1],tr[1][0],tr[1][1])           
                        elif preset == 2:
                            mm3 += convPtoR_2(tr[0][0],tr[0][1],tr[1][0],tr[1][1])          
                        mmm3 = mm3/ptn3
                        v3 = mmm3*fps*ms2kmh*6
                    if result_pg4 > 0:
                        ptn4 += 1
                        if preset == 1:
                            mm4 += convPtoR_1(tr[0][0],tr[0][1],tr[1][0],tr[1][1])           
                        elif preset == 2:
                            mm4 += convPtoR_2(tr[0][0],tr[0][1],tr[1][0],tr[1][1])          
                        mmm4 = mm4/ptn4
                        v4 = mmm4*fps*ms2kmh*6
                    if result_pg5 > 0:
                        ptn5 += 1
                        if preset == 1:
                            mm5 += convPtoR_1(tr[0][0],tr[0][1],tr[1][0],tr[1][1])           
                        elif preset == 2:
                            mm5 += convPtoR_2(tr[0][0],tr[0][1],tr[1][0],tr[1][1])          
                        mmm5 = mm5/ptn5
                        v5 = mmm5*fps*ms2kmh*6
                    if result_pg6 > 0:
                        ptn6 += 1
                        if preset == 1:
                            mm6 += convPtoR_1(tr[0][0],tr[0][1],tr[1][0],tr[1][1])           
                        elif preset == 2:
                            mm6 += convPtoR_2(tr[0][0],tr[0][1],tr[1][0],tr[1][1])          
                        mmm6 = mm6/ptn6
                        v6 = mmm6*fps*ms2kmh*6
        
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in of_track]:
            cv2.circle(mask, (x, y), 3, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                of_track.append([(x, y)])
        prev_gray = frame_gray

        ## swhan method
        if frame_idx % detect_interval == 0:
            if ptn1 > 0:
                avg_speed_lane_1 = v1
                prv1=v1
            elif ptn1 == 0:
                avg_speed_lane_1 = 0 
                prv1 = 0
            if ptn2 > 0:
                avg_speed_lane_2 = v2
                prv2=v2
            elif ptn2 == 0:
                avg_speed_lane_2 = 0 
                prv2 = 0
            if ptn3 > 0:
                avg_speed_lane_3 = v3
                prv3=v3
            elif ptn3 == 0:
                avg_speed_lane_3 = 0 
                prv3 = 0
            if ptn4 > 0:
                avg_speed_lane_4 = v4
                prv4=v4
            elif ptn4 == 0:
                avg_speed_lane_4 = 0 
                prv4 = 0
            if ptn5 > 0:
                avg_speed_lane_5 = v5
                prv5=v5
            elif ptn5 == 0:
                avg_speed_lane_5 = 0 
                prv5 = 0
            if ptn6 > 0:
                avg_speed_lane_6 = v6
                prv6=v6
            elif ptn6 == 0:
                avg_speed_lane_6 = 0 
                prv6 = 0
        else:
            avg_speed_lane_1 = prv1 
            avg_speed_lane_2 = prv2 
            avg_speed_lane_3 = prv3 
            avg_speed_lane_4 = prv4 
            avg_speed_lane_5 = prv5 
            avg_speed_lane_6 = prv6 
        
        ptn1, ptn2, ptn3, ptn4, ptn5, ptn6 = 0, 0, 0, 0, 0, 0

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        
        for det in detections:
            bbox = det.to_tlbr()
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            if len(class_names) > 0:
                class_name = class_names[0]
            
            i += 1
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            
            # global point matching
            track.matching_point[0] = center[0]
            track.matching_point[1] = center[1]
            
            temp_list = [track.matching_point[0],track.matching_point[1],frame_idx]

            if len(track.matching_point_list) == 4:
                track.matching_point_list.append(temp_list)

                x1 = track.matching_point_list[0][0]
                y1 = track.matching_point_list[0][1]
                x2 = track.matching_point_list[-1][0]
                y2 = track.matching_point_list[-1][1]
            
                # If the pixel don't change, the speed should be zero.
                if x1 == x2 and y1 == y2:
                    track.matching_point_list.pop(0)
                else:
                    time1 = track.matching_point_list[0][2]
                    time2 = track.matching_point_list[-1][2]
            
                    if preset == 1:
                        R_dist1 = convPtoR_1(x1,y1,x2,y2)
                        t_time1 = (time2-time1)/30
                        speed = int(3.6*R_dist1 //t_time1)
                        #print("time1 : ",time1, "time2 : ",time2)

                    elif preset == 2:
                        R_dist1 = convPtoR_2(x1,y1,x2,y2)
                        t_time1 = (time2-time1)/30
                        speed = int(3.6*R_dist1 //t_time1)
                        #print("time1 : ",time1, "time2 : ",time2)
                                       
                track.matching_point_list.pop(0)
                if frame_idx % 6 ==1 :
                    track.speed = speed
                cv2.putText(frame, str(int(track.speed))+'km/h', (int(bbox[0]), int(bbox[1]+((bbox[3]-bbox[1])/2))),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)
            elif len(track.matching_point_list) == 0 :
                track.matching_point_list.append(temp_list)
            elif track.matching_point_list[-1][0] != track.matching_point[0] and track.matching_point_list[-1][1] != track.matching_point[1]: 
                track.matching_point_list.append(temp_list)

            cv2.circle(frame, (track.matching_point[0], track.matching_point[1]), 5, (0,0,255),-1)
            
            # traffic lane by lane
            if cv2.pointPolygonTest(lane_1, center, False) >= 0:
                cnt_lane_1 += 1
                track.driving_lane = 1
            elif cv2.pointPolygonTest(lane_2, center, False) >= 0:
                cnt_lane_2 += 1 
                track.driving_lane = 2
            elif cv2.pointPolygonTest(lane_3, center, False) >= 0:
                cnt_lane_3 += 1
                track.driving_lane = 3
            elif cv2.pointPolygonTest(lane_4, center, False) >= 0:
                cnt_lane_4 += 1 
                track.driving_lane = 4
            elif cv2.pointPolygonTest(lane_5, center, False) >= 0:
                cnt_lane_5 += 1 
                track.driving_lane = 5
            elif cv2.pointPolygonTest(lane_6, center, False) >= 0:
                cnt_lane_6 += 1 
                track.driving_lane = 6

            cv2.putText(frame, "ID:"+str(track.track_id) + "/"+ str(track.driving_lane), (int(bbox[0]), int(bbox[1])-20),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            pts[track.track_id].append(center)
            ctime[track.time].append(time.time())

            thickness = 5
            
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

            #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)
 
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)

        cv2.putText(frame, "Traffic, Avg_speed", (570,40), 0, 1, (0,255,0),  2)
        cv2.putText(frame, "Lane_1: " + str(cnt_lane_1)+', '+str(int(avg_speed_lane_1)), (500,80), 0, 1, (0,255,0),  2)
        cv2.putText(frame, "Lane_2: " + str(cnt_lane_2)+', '+str(int(avg_speed_lane_2)), (500,120), 0, 1, (0,255,0),  2)
        cv2.putText(frame, "Lane_3: " + str(cnt_lane_3)+', '+str(int(avg_speed_lane_3)), (500,160), 0, 1, (0,255,0),  2)
        cv2.putText(frame, "Lane_4: " + str(cnt_lane_4)+', '+str(int(avg_speed_lane_4)), (500,200), 0, 1, (0,255,0),  2)
        cv2.putText(frame, "Lane_5: " + str(cnt_lane_5)+', '+str(int(avg_speed_lane_5)), (500,240), 0, 1, (0,255,0),  2)
        cv2.putText(frame, "Lane_6: " + str(cnt_lane_6)+', '+str(int(avg_speed_lane_6)), (500,280), 0, 1, (0,255,0),  2)
       
        frame_idx += 1 

        if writeVideo_flag:
            #save a frame
            # cv2.fillPoly(frame, [mask], (0,0,0))
            # for i,v in enumerate(global_point):            
            #     cv2.circle(frame, v, 1, (0,255,255),-1)
            cv2.addWeighted(cmask, alpha, frame, 1 - alpha, 0, frame)
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')                
            fps  = ( fps + (1./(time.time()-t1)) ) / 2

    print(" ")
    print("[Finish]")
    end = time.time()

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YoLo4())
