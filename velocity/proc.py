import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sys
from scipy.optimize import linear_sum_assignment
import pickle
def bool_to_uint8(mask):
    return 255*(mask).astype("uint8")

path_dir0 = pathlib.Path(sys.argv[1])
g_kksize = 11
l_kksize = 11

def lap(ch):
    lap_blue = cv2.Laplacian(ch, ksize=l_kksize, ddepth=cv2.CV_32F)
    lap_blue = np.clip(lap_blue/(2**(l_kksize*2 -2-2-2)), 0 , 10)
    # lap_blue = cv2.morphologyEx(lap_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    return lap_blue

def detect_cnts(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # back_cnt
    
    return [ cnt for cnt in contours if  60 < cv2.contourArea(cnt) < 20000]
def nt_(a):
    return a.round().astype(int)
# def proc_cnts(cnts):
#     areas = [cv2.contourArea(cnt) for cnt in cnts]

#     cnt_a_s = sorted(zip(cnts, areas), key=lambda x: x[1])

def mp(cnt):
    M = cv2.moments(cnt)
    # print(cnt.shape)
    x = (M["m10"] / (M["m00"]+1e-10))
    y = (M["m01"] / (M["m00"] + 1e-10))
    return np.array([x,y])



def proc(img):
    output = []
    # crop = img[track_point[1]:blink_point[1], track_point[0]:blink_point[0], :]
    # dst = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
    # dst = img
    
    dst = cv2.medianBlur(img, 19)
    cv2.imshow("img", img)
    cv2.imshow("dst", dst)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    blue = dst[:, :, 0].astype(np.float32)/255.
    red = dst[:, :, 2].astype(np.float32)/255.
    th_sd = 0.2

    # blue /= blue.max()
    # red /= red.max()
    
    blue = np.clip(blue - blue[blue < th_sd].mean(), 0, 1)
    red = np.clip(red - red[red < th_sd].mean(), 0, 1)
    
    blue /= blue.max()
    red /= red.max()
    

    cv2.imshow("red", red)
    cv2.imshow("blue", blue)
    # lap_blue = lap(blue)
    # lap_red = lap(red)
    # dx = cv2.Sobel(blue, cv2.CV_32F, 1, 0, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
    # dy = cv2.Sobel(blue, cv2.CV_32F, 0, 1, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
    # sobel_abs = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    
    # cv2.imshow("sobel_abs", sobel_abs*10)
    # cv2.imshow("lap", lap)
    th_l = 0.25
    
    red_less_mask = bool_to_uint8(red < th_l)
    blue_less_mask = bool_to_uint8(blue < th_l)
    th = 0.05
    # red_lap_mask = bool_to_uint8(lap_red > th)
    # blue_lap_mask = bool_to_uint8(lap_blue > th)
    

    less_comb = np.zeros_like(img)
    less_comb[:, :, 0] = blue_less_mask
    less_comb[:, :, 2] = red_less_mask
    cv2.imshow("less_comb", less_comb)
    # lap_mask_comb = np.zeros_like(img)
    # lap_mask_comb[:, :, 0] = blue_lap_mask
    # lap_mask_comb[:, :, 2] = red_lap_mask
    # cv2.imshow("lap_mask_comb", lap_mask_comb)
    # lap_comb = np.zeros_like(img)
    m = 10
    # lap_comb[:, :, 0] = np.clip(lap_blue*255*m, 0, 255).astype("uint8")
    # lap_comb[:, :, 2] = np.clip(lap_red*255*m, 0, 255).astype("uint8")
    # cv2.imshow("lap_comb", lap_comb)

    cnts_blue = detect_cnts(blue_less_mask)
    cnts_red = detect_cnts(red_less_mask)
    debug_cnts = np.zeros_like(img)
    cv2.drawContours(debug_cnts, cnts_red, -1, (0,0,255), 2)
    cv2.drawContours(debug_cnts, cnts_blue, -1, (255,0,0), 2)
   

    mat = np.zeros((len(cnts_blue), len(cnts_red)))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j] = cv2.matchShapes(cnts_blue[i],cnts_red[j],1,0.0)

    mps_blue = [mp(cnt) for cnt in cnts_blue ]
    mps_red = [mp(cnt) for cnt in cnts_red ]
    areas_blue = [cv2.contourArea(cnt) for cnt in cnts_blue]
    areas_red = [cv2.contourArea(cnt) for cnt in cnts_red]
    
    d = np.zeros((len(cnts_blue), len(cnts_red)))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            d[i, j] = np.linalg.norm(mps_red[j] - mps_blue[i])
    # seems_to_be_an_asssigne = []
    th_d = 50
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if  d[i, j] < th_d and mat[i, j] < 0.5  and abs(areas_blue[i] - areas_red[j])*2 / (areas_blue[i]+areas_red[j]) < 0.9:
                # and np.dot(mps_blue[i] - mps_red[j], np.array([np.sqrt(1/2), np.sqrt(1/2)]))/d[i, j] > 0.5:
                # cv2.line(debug_cnts, nt_(mps_red[j]), nt_(mps_blue[i]), (0, 255, 255), 2)
                # seems_to_be_an_asssigne.append((i, j))
                pass
            else:
                d[i, j] = 100000000
    i_to_del = []
    j_to_del = []
    indexes = np.zeros((len(cnts_blue), len(cnts_red), 2), dtype=int)
    print(indexes.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            indexes[i, j, 0] = i
            indexes[i, j, 1] = j
    for i in range(mat.shape[0]):
        if d[i, :].min() >= th_d:
            i_to_del.append(i)
    print(i_to_del, j_to_del)
    # j_to_del = []
    for j in range(mat.shape[1]):
        # print()
        if d[:, j].min() >= th_d:
            j_to_del.append(j)
    indexes = np.delete(indexes, i_to_del, axis=0)
    indexes = np.delete(indexes, j_to_del, axis=1)
    print(indexes.shape)
    # print(seems_to_be_an_asssigne)      
    d_t = np.zeros(indexes.shape[:2])
    for i in range(indexes.shape[0]):     
        for j in range(indexes.shape[1]):
            d_t[i, j] = d[indexes[i, j][0],  indexes[i, j][1]]
    row_ind, col_ind = linear_sum_assignment(d_t)   

    for i_, j_ in zip(row_ind, col_ind):
        i, j = indexes[i_, j_]
        if d[i, j] < th_d:
            cnt_blue = cnts_blue[i]
            cnt_red = cnts_red[j]
            item = [mps_blue[i]- mps_red[j]]
            for k in [cnt_blue, cnt_red]:
                x,y,w,h = cv2.boundingRect(k)
                R = np.sqrt(w*h)/2
                item.append(R)
                cv2.rectangle(debug_cnts,(x,y),(x+w,y+h),(0,255,0),2)
            output.append(item)
            cv2.line(debug_cnts, nt_(mps_red[j]), nt_(mps_blue[i]), (255, 255, 0), 2)

    cv2.imshow("debug_cnts", debug_cnts)
    print(output)
    return output
    # flow = cv2.calcOpticalFlowFarneback(red_less_mask, blue_less_mask, None, 0.5, 3, 15, 5, 5, 1.2, 0)
    # hsv = np.zeros_like(img)
    # hsv[..., 1] = 255
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('frame2', bgr)
    # print(mat)
    # # cv2.imshow("mat", mat)
    # plt.figure()
    # plt.imshow(mat)
    # # plt.show()
    # plt.figure()
    # plt.imshow(d)
    # plt.show()
    # cv2.imshow("red_less_mask", red_less_mask)
    # cv2.imshow("blue_less_mask", blue_less_mask)
# 

for path_dir in path_dir0.glob("*"):
    output_all = []

    for p in  path_dir.glob("*.jpg"):
        print(p)
        try:
            img = cv2.imread(str(p))
            img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)
            output = proc(img)
            output_all.extend(output)
            pickle.dump(output_all, open(path_dir / "output_all.pickle", "wb"))
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        except:
            print("Exception")

    