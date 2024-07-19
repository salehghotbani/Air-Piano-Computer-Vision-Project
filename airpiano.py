import cv2 as cv
import numpy as np
import pyautogui
import math


def press(key):
    pyautogui.press(key)


# https://www.onlinepianist.com/virtual-piano

cap = cv.VideoCapture(0)


# cap.set(3,600)
# cap.set(4,800)


# cv.namedWindow('Trackbar')
# cv.createTrackbar("L-H","Trackbar",0,255,nothing)
# cv.createTrackbar("L-S","Trackbar",0,255,nothing)
# cv.createTrackbar("L-V","Trackbar",0,255,nothing)
# cv.createTrackbar("U-H","Trackbar",255,255,nothing)
# cv.createTrackbar("U-S","Trackbar",255,255,nothing)
# cv.createTrackbar("U-V","Trackbar",255,255,nothing)

while True:
    _, frame = cap.read()
    frame = cv.resize(frame, (580, 600))
    frame = cv.flip(frame, 1)
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    # frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # l_h=cv.getTrackbarPos("L-H", "Trackbar")
    # l_s=cv.getTrackbarPos("L-S", "Trackbar")
    # l_v=cv.getTrackbarPos("L-V", "Trackbar")
    # u_h=cv.getTrackbarPos("U-H", "Trackbar")
    # u_s=cv.getTrackbarPos("U-S", "Trackbar")
    # u_v=cv.getTrackbarPos("U-V", "Trackbar")

    # lower_black=np.array([l_h,l_s,l_v]) # get proper values from experimenting with trackbar.
    # upper_black=np.array([u_h,u_s,u_v]) 

    lower_black = np.array([0, 0, 0])  # get the proper values from experimenting with trackbar.
    upper_black = np.array([255, 255, 50])
    mask = cv.inRange(hsv, lower_black, upper_black)

    cv.rectangle(frame, (5, 0), (40, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (43, 0), (75, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (78, 0), (105, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (108, 0), (140, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (143, 0), (175, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (178, 0), (205, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (208, 0), (240, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (243, 0), (275, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (278, 0), (305, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (308, 0), (340, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (343, 0), (375, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (378, 0), (405, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (408, 0), (440, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (443, 0), (475, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (478, 0), (505, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (508, 0), (540, 250), (255, 255, 255), 1)
    cv.rectangle(frame, (543, 0), (575, 250), (255, 255, 255), 1)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        if 14000 < cv.contourArea(contour) < 26000:
            # print(cv.contourArea(contour))
            cv.drawContours(frame, contour, -1, (0, 0, 255), 3)

            hull = cv.convexHull(contour, returnPoints=False)
            defects = cv.convexityDefects(contour, hull)

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                x, y, p, q = start[0], start[1], far[0], far[1]
                dist = math.sqrt((x - p) ** 2 + (y - q) ** 2)

                # cv.line(frame,start,end,[0,255,0],2) draws the convex hull polygon.
                if dist >= 50:
                    cv.circle(frame, start, 5, [20, 240, 255], -1)  # draws the finger tips.

                if 5 < p < 40 and 0 < q < 250:
                    press('Q')
                    break
                if 43 < p < 75 and 0 < q < 250:
                    press('W')
                    break
                if 78 < p < 105 and 0 < q < 250:
                    press('E')
                    break
                if 108 < p < 140 and 0 < q < 250:
                    press('R')
                    break
                if 143 < p < 175 and 0 < q < 250:
                    press('T')
                    break
                if 178 < p < 205 and 0 < q < 250:
                    press('Y')
                    break
                if 208 < p < 240 and 0 < q < 250:
                    press('U')
                    break
                if 243 < p < 275 and 0 < q < 250:
                    press('I')
                    break
                if 278 < p < 305 and 0 < q < 250:
                    press('O')
                    break
                if 308 < p < 340 and 0 < q < 250:
                    press('P')
                    break
                if 343 < p < 375 and 0 < q < 250:
                    press('Z')
                    break
                if 378 < p < 405 and 0 < q < 250:
                    press('X')
                    break
                if 408 < p < 440 and 0 < q < 250:
                    press('C')
                    break
                if 443 < p < 475 and 0 < q < 250:
                    press('V')
                    break
                if 478 < p < 505 and 0 < q < 250:
                    press('B')
                    break
                if 508 < p < 540 and 0 < q < 250:
                    press('N')
                    break
                if 543 < p < 575 and 0 < q < 250:
                    press('M')
                    break

    cv.imshow('frame', frame)
    # cv.imshow('mask',mask)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
