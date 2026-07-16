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


# Максимум точек на один тайл
MAX_FEATURES = 3000


# Размер ячейки пространственного фильтра
# 300 px = 60 метров при 0.2 м/px
CELL_SIZE = 300


# Максимум точек в одной ячейке
MAX_POINTS_CELL = 10


# Минимальный размер ключевой точки
MIN_SIZE = 7


# Минимальное количество повторений точки
MIN_REPEAT = 3


# Масштабы для поиска
SCALES = [
    0.5,
    0.75,
    1.0,
    1.25
]


# Количество процессов
# None = автоматически
WORKERS = 5



# =====================================================
# Служебные функции
# =====================================================


def save_keypoints(path, keypoints):

    data = []

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



def point_distance(a, b):

    dx = a.pt[0] - b.pt[0]
    dy = a.pt[1] - b.pt[1]

    return (
        dx * dx +
        dy * dy
    ) ** 0.5



def repeatability_filter(points, descriptors):

    """
    Оставляет только точки,
    которые появились минимум MIN_REPEAT раз
    """

    result_points = []
    result_desc = []

    used = set()


    for i, p in enumerate(points):

        if i in used:
            continue


        group = [i]


        used.add(i)


        for j, q in enumerate(points):

            if j in used:
                continue


            if point_distance(p, q) < 8:

                group.append(j)
                used.add(j)


        if len(group) >= MIN_REPEAT:

            best = max(
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



def spatial_filter(points, descriptors):

    """
    Ограничение количества точек
    в одной области
    """

    cells = {}

    out_points = []
    out_desc = []


    indexes = sorted(
        range(len(points)),
        key=lambda i:
        points[i].response *
        points[i].size,
        reverse=True
    )


    for i in indexes:

        p = points[i]


        cell = (

            int(
                p.pt[0] //
                CELL_SIZE
            ),

            int(
                p.pt[1] //
                CELL_SIZE
            )

        )


        if cells.get(cell, 0) >= MAX_POINTS_CELL:
            continue


        cells[cell] = (
            cells.get(cell, 0)
            + 1
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
# Обработка одного тайла
# =====================================================


def process_tile(filename):


    try:

        image_path = os.path.join(
            TILES_DIR,
            filename
        )


        name = filename.replace(
            ".png",
            ""
        )


        output = os.path.join(
            OUTPUT_DIR,
            name
        )


        os.makedirs(
            output,
            exist_ok=True
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )


        if image is None:
            return filename



        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        all_points = []
        all_desc = []



        # =====================================
        # Multi-scale AKAZE
        # =====================================


        for scale in SCALES:


            resized = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )


            detector = cv2.AKAZE_create(
                descriptor_type=
                cv2.AKAZE_DESCRIPTOR_MLDB,

                threshold=0.001
            )


            kp, desc = (
                detector.detectAndCompute(
                    resized,
                    None
                )
            )


            if desc is None:
                continue



            for p in kp:

                p.pt = (

                    p.pt[0] / scale,

                    p.pt[1] / scale

                )



            for i, p in enumerate(kp):

                if p.size >= MIN_SIZE:

                    all_points.append(p)

                    all_desc.append(
                        desc[i]
                    )



        if len(all_points) == 0:
            return filename



        all_desc = np.array(
            all_desc
        )



        # =====================================
        # Повторяемость
        # =====================================

        points, desc = (
            repeatability_filter(
                all_points,
                all_desc
            )
        )


        if len(points) == 0:
            return filename



        # =====================================
        # Пространственный фильтр
        # =====================================

        points, desc = (
            spatial_filter(
                points,
                desc
            )
        )



        # =====================================
        # Ограничение количества
        # =====================================

        if len(points) > MAX_FEATURES:


            indexes = sorted(
                range(len(points)),
                key=lambda i:
                points[i].response,
                reverse=True
            )[:MAX_FEATURES]


            points = [
                points[i]
                for i in indexes
            ]


            desc = desc[indexes]



        # =====================================
        # Сохранение
        # =====================================

        save_keypoints(
            os.path.join(
                output,
                "keypoints.npy"
            ),
            points
        )


        np.save(
            os.path.join(
                output,
                "descriptors.npy"
            ),
            desc
        )



        marked = cv2.drawKeypoints(
            image,
            points,
            None,
            flags=
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )


        cv2.imwrite(
            os.path.join(
                output,
                "features.png"
            ),
            marked
        )



        with open(
            os.path.join(
                output,
                "metadata.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:


            json.dump(
                {
                    "tile":
                        filename,

                    "points":
                        len(points),

                    "method":
                        "AKAZE multiscale repeatability",

                    "scales":
                        SCALES
                },

                f,

                indent=4,

                ensure_ascii=False
            )



        return filename



    except Exception as e:

        print(
            "Ошибка:",
            filename,
            e
        )

        return filename



# =====================================================
# Главная функция
# =====================================================


def main():


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    files = [

        f
        for f in os.listdir(TILES_DIR)

        if f.endswith(".png")

    ]


    if WORKERS is None:

        workers = max(
            1,
            cpu_count() - 1
        )

    else:

        workers = WORKERS



    print(
        "Тайлов:",
        len(files)
    )

    print(
        "Процессов:",
        workers
    )



    with Pool(
        processes=workers
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
        "Обработка завершена"
    )



if __name__ == "__main__":

    main()