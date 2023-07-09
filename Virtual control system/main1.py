import cv2
import mediapipe as mp
import time
import pyautogui
import python_class1 as htm
from selenium import webdriver
import autopy
import numpy as np
import mouse


def keyboard():
    text = ""

    pyautogui.keyDown('ctrl')
    pyautogui.keyDown('shift')
    pyautogui.press('o')
    pyautogui.keyUp('ctrl')
    pyautogui.keyUp('shift')

    text = "Keyboard out"
    return text


def up_down(lmList):
    text = "up_down mode"
    if len(lmList) != 0:
        fing = detector.fingersUp()
        print(fing)
        if fing[1] == 1 and fing[2] == 1:
            browser.execute_script("window.scrollTo( window.scrollY ,window.scrollY+30)")
        elif fing[1] == 1:
            browser.execute_script("window.scrollTo(window.scrollY ,window.scrollY-30)")
    text = "up down mode"

    return text


def zoom(img, lmList):
    text = "zoom mode"
    k = 50
    if len(lmList) != 0:
        length, img, lineInfo = detector.findDistance(4, 8, img)
        print(length)
        if length > 170:
            length = 170
        elif length < 20:
            length = 20

        length = (length / 170) * 100
        k = k + length

        browser.execute_script("document.body.style.zoom='{}%'".format(k))
    return text


def mouse1(img, lmList, plocX, plocY, fingers):
    text = "mouse mode"
    if (fingers[3] == 0 and fingers[4] == 0):
        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[4][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            # print(fingers)
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)
            # 4. Only Index Finger : Moving Mode
            if fingers[1] == 1:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[1] == 0:
                # 9. Find distance between fingers
                length, img, lineInfo = detector.findDistance(4, 8, img)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    mouse.click("left")
    elif (fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0):
        text = keyboard()
    return text, plocX, plocY


wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

wCam, hCam = 640, 480

cTime = 0
ct = 0
text = "normal mode"

browser = webdriver.Chrome()
browser.get("https://www.linkedin.com/")
f = 0
b = 1
while True:

    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList, bbox = detector.findPosition(img)
    if len(lmList) > 0:

        fing = detector.fingersUp()
        print(fing)
        print((np.sum(fing) == 5 and b == 1))
        print(b)
        if (np.sum(fing) == 5 and b == 1):
            if f == 3:
                f = -1
                text = "normal mode"
            f = f + 1

        if f == 1:
            text, plocX, plocY = mouse1(img, lmList, plocX, plocY, fing)

        elif f == 2:
            text = up_down(lmList)

        elif f == 3:
            text = zoom(img, lmList)

        if np.sum(fing) == 5:
            b = 0
        elif np.sum(fing) == 0:
            b = 1

    cv2.putText(img, text, (80, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 255), 3)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)