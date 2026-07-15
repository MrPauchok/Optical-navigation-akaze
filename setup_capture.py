import cv2
import json
import pyautogui


import numpy as np

WIDTH=1280
HEIGHT=720


x=100
y=100


drag=False
old_x=0
old_y=0



def mouse(event, mx,my,flags,param):

    global x,y
    global drag
    global old_x,old_y


    if event==cv2.EVENT_LBUTTONDOWN:

        drag=True

        old_x=mx
        old_y=my



    elif event==cv2.EVENT_MOUSEMOVE and drag:

        dx=mx-old_x
        dy=my-old_y


        x+=dx
        y+=dy


        old_x=mx
        old_y=my



    elif event==cv2.EVENT_LBUTTONUP:

        drag=False





cv2.namedWindow(
    "capture area"
)


cv2.setMouseCallback(
    "capture area",
    mouse
)



while True:


    screen=pyautogui.screenshot()


    frame=cv2.cvtColor(
        np.array(screen),
        cv2.COLOR_RGB2BGR
    )


    h,w=frame.shape[:2]


    overlay=frame.copy()



    cv2.rectangle(

        overlay,

        (x,y),

        (
            x+WIDTH,
            y+HEIGHT
        ),

        (0,255,0),

        3
    )



    cv2.imshow(
        "capture area",
        overlay
    )



    key=cv2.waitKey(20)



    if key==13: # ENTER


        with open(
            "capture.json",
            "w"
        ) as f:


            json.dump(

                {
                    "left":x,
                    "top":y
                },

                f,
                indent=4
            )


        break



    if key==27:

        break



cv2.destroyAllWindows()