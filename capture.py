import cv2
import mss
import numpy as np
import json
import os


CONFIG = "capture.json"


class ScreenCapture:


    def __init__(self):

        self.sct = mss.mss()


        self.load_position()



    def load_position(self):

        if os.path.exists(CONFIG):

            with open(CONFIG,"r") as f:

                pos=json.load(f)


        else:

            pos={
                "left":0,
                "top":0
            }


        self.monitor={

            "left":pos["left"],
            "top":pos["top"],

            "width":1280,
            "height":720

        }



    def read(self):

        img=self.sct.grab(
            self.monitor
        )


        frame=np.array(img)


        frame=cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )


        return frame