import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# =====================================================
# Настройки
# =====================================================

TILES_DIR = "tiles"
OUTPUT_DIR = "features"


MAX_FEATURES = 3000


# 300 px = 60 метров
CELL_SIZE = 300


MAX_POINTS_CELL = 10


MIN_SIZE = 7


# точка должна появиться
# минимум в 3 масштабах
MIN_REPEAT = 3


SCALES = [
    0.5,
    0.75,
    1.0,
    1.25
]


WORKERS = 5



# =====================================================
# Зеленая маска
# =====================================================

GREEN_LOW = np.array(
    [25, 40, 20]
)

GREEN_HIGH = np.array(
    [95, 255, 255]
)



def remove_green_areas(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    mask = cv2.inRange(
        hsv,
        GREEN_LOW,
        GREEN_HIGH
    )


    kernel = np.ones(
        (15,15),
        np.uint8
    )


    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )


    result = image.copy()


    result[
        mask > 0
    ] = 0


    return result



# =====================================================
# Бинаризация
# =====================================================

def create_binary_image(gray):


    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )


    binary = cv2.adaptiveThreshold(
        blur,
        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        31,

        5
    )


    kernel = np.ones(
        (3,3),
        np.uint8
    )


    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    return binary



# =====================================================
# Сохранение точек
# =====================================================

def save_keypoints(path, keypoints):

    data=[]


    for kp in keypoints:

        data.append(
            [
                kp.pt[0],
                kp.pt[1],
                kp.size,
                kp.angle,
                kp.response,
                kp.octave
            ]
        )


    np.save(
        path,
        np.array(data)
    )



# =====================================================
# Расстояние
# =====================================================

def point_distance(a,b):

    dx=a.pt[0]-b.pt[0]
    dy=a.pt[1]-b.pt[1]


    return (
        dx*dx+
        dy*dy
    )**0.5



# =====================================================
# Repeatability
# =====================================================

def repeatability_filter(points, descriptors):


    result_points=[]
    result_desc=[]


    used=set()



    for i,p in enumerate(points):


        if i in used:
            continue


        group=[i]

        used.add(i)



        for j,q in enumerate(points):


            if j in used:
                continue


            if point_distance(p,q)<8:


                group.append(j)

                used.add(j)



        if len(group)>=MIN_REPEAT:


            best=max(
                group,
                key=lambda x:
                points[x].response
            )


            result_points.append(
                points[best]
            )


            result_desc.append(
                descriptors[best]
            )



    return (
        result_points,
        np.array(result_desc)
    )



# =====================================================
# Пространственный фильтр
# =====================================================

def spatial_filter(points, descriptors):


    cells={}


    out_points=[]
    out_desc=[]



    indexes=sorted(
        range(len(points)),

        key=lambda i:
        points[i].response*
        points[i].size,

        reverse=True
    )



    for i in indexes:


        p=points[i]


        cell=(

            int(
                p.pt[0]//
                CELL_SIZE
            ),

            int(
                p.pt[1]//
                CELL_SIZE
            )

        )



        if cells.get(cell,0)>=MAX_POINTS_CELL:
            continue



        cells[cell]=(
            cells.get(cell,0)+1
        )


        out_points.append(p)

        out_desc.append(
            descriptors[i]
        )



    return (
        out_points,
        np.array(out_desc)
    )



# =====================================================
# Обработка тайла
# =====================================================

def process_tile(filename):


    try:


        path=os.path.join(
            TILES_DIR,
            filename
        )


        name=filename[:-4]


        out=os.path.join(
            OUTPUT_DIR,
            name
        )


        os.makedirs(
            out,
            exist_ok=True
        )



        image=cv2.imread(
            path,
            cv2.IMREAD_COLOR
        )


        if image is None:
            return filename



        # удаление зелени

        filtered=remove_green_areas(
            image
        )


        cv2.imwrite(
            os.path.join(
                out,
                "filtered.png"
            ),
            filtered
        )



        gray=cv2.cvtColor(
            filtered,
            cv2.COLOR_BGR2GRAY
        )



        binary=create_binary_image(
            gray
        )


        cv2.imwrite(
            os.path.join(
                out,
                "binary.png"
            ),
            binary
        )



        points_all=[]
        desc_all=[]



        detector=cv2.AKAZE_create(
            descriptor_type=
            cv2.AKAZE_DESCRIPTOR_MLDB,

            threshold=0.001
        )



        for scale in SCALES:


            gray_scaled=cv2.resize(
                gray,
                None,

                fx=scale,
                fy=scale,

                interpolation=cv2.INTER_AREA
            )



            binary_scaled=cv2.resize(
                binary,
                None,

                fx=scale,
                fy=scale,

                interpolation=cv2.INTER_NEAREST
            )



            kp=detector.detect(
                binary_scaled,
                None
            )


            kp,desc=detector.compute(
                gray_scaled,
                kp
            )


            if desc is None:
                continue



            for p in kp:


                p.pt=(

                    p.pt[0]/scale,

                    p.pt[1]/scale

                )



            for i,p in enumerate(kp):


                if p.size>=MIN_SIZE:


                    points_all.append(p)

                    desc_all.append(
                        desc[i]
                    )



        if len(points_all)==0:
            return filename



        desc_all=np.array(
            desc_all
        )



        points,desc=(
            repeatability_filter(
                points_all,
                desc_all
            )
        )



        if len(points)==0:
            return filename



        points,desc=(
            spatial_filter(
                points,
                desc
            )
        )



        if len(points)>MAX_FEATURES:


            idx=sorted(
                range(len(points)),

                key=lambda i:
                points[i].response,

                reverse=True

            )[:MAX_FEATURES]


            points=[
                points[i]
                for i in idx
            ]


            desc=desc[idx]



        save_keypoints(
            os.path.join(
                out,
                "keypoints.npy"
            ),
            points
        )


        np.save(
            os.path.join(
                out,
                "descriptors.npy"
            ),
            desc
        )



        marked=cv2.drawKeypoints(
            gray,
            points,
            None,

            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )


        cv2.imwrite(
            os.path.join(
                out,
                "features.png"
            ),
            marked
        )



        with open(
            os.path.join(
                out,
                "metadata.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:


            json.dump(

                {
                    "tile":filename,

                    "points":len(points),

                    "method":
                    "AKAZE binary detect + gray descriptor",

                    "scales":SCALES
                },

                f,

                indent=4,

                ensure_ascii=False
            )


        return filename



    except Exception as e:


        print(
            "Ошибка",
            filename,
            e
        )


        return filename



# =====================================================
# Запуск
# =====================================================

def main():


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    files=[

        f for f in os.listdir(TILES_DIR)

        if f.lower().endswith(".png")

    ]



    print(
        "Тайлов:",
        len(files)
    )


    print(
        "Процессов:",
        WORKERS
    )



    with Pool(
        processes=WORKERS
    ) as pool:


        list(
            tqdm(
                pool.imap(
                    process_tile,
                    files
                ),

                total=len(files)
            )
        )


    print(
        "Готово"
    )



if __name__=="__main__":

    main()