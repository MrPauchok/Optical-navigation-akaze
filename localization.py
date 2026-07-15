import os
import json
import cv2
import numpy as np

from multiprocessing import Pool


# =====================

TILE_MAP = "tile_map.json"
TILE_COORDS = "tile_coords.json"

START_TILE = "0_0"

SEARCH_RADIUS = 1

MIN_MATCHES = 15
MIN_INLIERS = 10

WORKERS = 2


# =====================
# функция рабочего процесса
# =====================


def match_tile(args):

    cam_desc, tile_desc, index = args


    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )


    matches = matcher.knnMatch(
        cam_desc,
        tile_desc,
        k=2
    )


    good = []


    for m, n in matches:


        if m.distance < 0.75 * n.distance:


            # только простые данные
            good.append(
                (
                    m.queryIdx,
                    m.trainIdx,
                    float(m.distance)
                )
            )


    return index, good



# =====================


class Localizer:


    def __init__(self, features_dir):


        self.features_dir = features_dir


        self.tiles = {}

        self.tile_map = {}

        self.coords = {}


        self.current_index = START_TILE



        self.akaze = cv2.AKAZE_create(
            threshold=0.001
        )


        self.pool = Pool(
            processes=WORKERS
        )


        self.load_files()

        self.load_features()



    # ------------------


    def load_files(self):


        with open(
            TILE_MAP,
            encoding="utf-8"
        ) as f:

            self.tile_map = json.load(f)



        with open(
            TILE_COORDS,
            encoding="utf-8"
        ) as f:

            self.coords = json.load(f)



    # ------------------


    def load_features(self):


        for index, name in self.tile_map.items():


            folder = os.path.join(
                self.features_dir,
                name
            )


            desc_file = os.path.join(
                folder,
                "descriptors.npy"
            )


            kp_file = os.path.join(
                folder,
                "keypoints.npy"
            )


            if not os.path.exists(desc_file):

                continue



            self.tiles[index] = {


                "name": name,


                "desc":
                    np.load(desc_file),


                "kp":
                    self.load_keypoints(
                        kp_file
                    )

            }



        print(
            "Загружено тайлов:",
            len(self.tiles)
        )



    # ------------------


    def load_keypoints(self, path):


        data = np.load(path)


        result = []


        for p in data:


            result.append(

                cv2.KeyPoint(

                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(p[3]),
                    float(p[4])

                )

            )


        return result



    # ------------------


    def neighbors(self):


        x, y = map(
            int,
            self.current_index.split("_")
        )


        result = []


        for dx in range(
            -SEARCH_RADIUS,
            SEARCH_RADIUS+1
        ):


            for dy in range(
                -SEARCH_RADIUS,
                SEARCH_RADIUS+1
            ):


                key = f"{x+dx}_{y+dy}"


                if key in self.tiles:

                    result.append(key)



        return result



    # ------------------


    def pixel_to_world(
            self,
            tile,
            px,
            py):


        data = self.coords[tile]


        X = (

            data["x0"]

            +

            px * data["pixel_size"]

        )


        Y = (

            data["y0"]

            -

            py * data["pixel_size"]

        )


        return X, Y



    # ------------------


    def process(self, frame):


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        kp, desc = self.akaze.detectAndCompute(
            gray,
            None
        )


        if desc is None:

            return None



        indexes = self.neighbors()


        tasks = []


        for index in indexes:


            tasks.append(

                (

                    desc,

                    self.tiles[index]["desc"],

                    index

                )

            )



        results = self.pool.map(
            match_tile,
            tasks
        )



        best = None



        for index, matches in results:


            if (

                best is None

                or

                len(matches)
                >
                len(best["matches"])

            ):


                best = {


                    "index": index,


                    "tile":
                        self.tiles[index],


                    "matches":
                        matches

                }



        if best is None:

            return None



        if len(best["matches"]) < MIN_MATCHES:

            return None



        src = []

        dst = []



        for m in best["matches"]:


            src.append(

                kp[m[0]].pt

            )


            dst.append(

                best["tile"]["kp"]
                [m[1]]
                .pt

            )



        H, mask = cv2.findHomography(

            np.float32(src),

            np.float32(dst),

            cv2.RANSAC,

            5

        )



        if H is None:

            return None



        if np.sum(mask) < MIN_INLIERS:

            return None



        h, w = gray.shape


        center = np.array(

            [

                [

                    [w/2, h/2]

                ]

            ],

            dtype=np.float32

        )



        point = cv2.perspectiveTransform(

            center,

            H

        )



        px, py = point[0][0]



        if best["index"] != self.current_index:


            print(

                "Переход:",

                self.current_index,

                "->",

                best["index"]

            )


            self.current_index = best["index"]



        X, Y = self.pixel_to_world(

            best["tile"]["name"],

            px,

            py

        )



        return (

            best["tile"]["name"],

            X,

            Y

        )



    # ------------------


    def close(self):


        self.pool.close()

        self.pool.join()