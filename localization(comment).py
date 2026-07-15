import os
import json
import cv2
import numpy as np

from multiprocessing import Pool


# =====================
# Файлы с описанием карты и координатами тайлов

TILE_MAP = "tile_map.json"
TILE_COORDS = "tile_coords.json"

# Начальный тайл, с которого начинается поиск

START_TILE = "0_0"

# Радиус поиска соседних тайлов относительно текущего положения

SEARCH_RADIUS = 1

# Минимальное количество найденных совпадений и внутренних совпадений

MIN_MATCHES = 15
MIN_INLIERS = 10

# Количество процессов параллельного поиска

WORKERS = 2


# =====================
# Функция рабочего процесса
# Выполняет поиск совпадений между кадром камеры и одним тайлом


def match_tile(args):

    # Получение дескрипторов изображения камеры,
    # дескрипторов тайла и его индекса

    cam_desc, tile_desc, index = args


    # Создание бинарного сопоставителя признаков AKAZE

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )


    # Поиск двух ближайших совпадений для каждого дескриптора

    matches = matcher.knnMatch(
        cam_desc,
        tile_desc,
        k=2
    )


    good = []


    # Фильтрация совпадений по коэффициенту Лоу

    for m, n in matches:


        # Отбрасываются слабые совпадения

        if m.distance < 0.75 * n.distance:


            # Сохраняются индексы точек и расстояние между дескрипторами

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


    # Инициализация локализатора

    def __init__(self, features_dir):


        # Папка с сохраненными признаками тайлов

        self.features_dir = features_dir


        # Словарь загруженных тайлов

        self.tiles = {}

        # Карта соответствия индексов и имен тайлов

        self.tile_map = {}

        # Координаты тайлов

        self.coords = {}


        # Текущий тайл локализации

        self.current_index = START_TILE



        # Создание детектора AKAZE

        self.akaze = cv2.AKAZE_create(
            threshold=0.001
        )


        # Создание пула процессов для параллельного поиска

        self.pool = Pool(
            processes=WORKERS
        )


        # Загрузка описаний карты

        self.load_files()

        # Загрузка признаков всех тайлов

        self.load_features()



    # ------------------


    # Загрузка файлов карты и координат

    def load_files(self):


        # Загрузка таблицы тайлов

        with open(
            TILE_MAP,
            encoding="utf-8"
        ) as f:

            self.tile_map = json.load(f)


        # Загрузка координат тайлов

        with open(
            TILE_COORDS,
            encoding="utf-8"
        ) as f:

            self.coords = json.load(f)



    # ------------------


    # Загрузка дескрипторов и ключевых точек тайлов

    def load_features(self):


        # Перебор всех тайлов карты

        for index, name in self.tile_map.items():


            # Формирование пути к папке тайла

            folder = os.path.join(
                self.features_dir,
                name
            )


            # Файлы с сохраненными признаками

            desc_file = os.path.join(
                folder,
                "descriptors.npy"
            )


            kp_file = os.path.join(
                folder,
                "keypoints.npy"
            )


            # Пропуск тайла без дескрипторов

            if not os.path.exists(desc_file):

                continue



            # Сохранение информации о тайле

            self.tiles[index] = {


                "name": name,


                # Загрузка дескрипторов AKAZE

                "desc":
                    np.load(desc_file),


                # Загрузка координат ключевых точек

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


    # Преобразование сохраненных координат точек
    # в объекты OpenCV KeyPoint

    def load_keypoints(self, path):


        data = np.load(path)


        result = []


        # Создание списка ключевых точек

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


    # Получение списка тайлов вокруг текущего положения

    def neighbors(self):


        # Получение координат текущего тайла

        x, y = map(
            int,
            self.current_index.split("_")
        )


        result = []


        # Проверка всех тайлов в заданном радиусе

        for dx in range(
            -SEARCH_RADIUS,
            SEARCH_RADIUS+1
        ):


            for dy in range(
                -SEARCH_RADIUS,
                SEARCH_RADIUS+1
            ):


                key = f"{x+dx}_{y+dy}"


                # Добавление существующих тайлов

                if key in self.tiles:

                    result.append(key)



        return result



    # ------------------


    # Перевод координат пикселя тайла
    # в мировые координаты карты

    def pixel_to_world(
            self,
            tile,
            px,
            py):


        data = self.coords[tile]


        # Расчет координаты X

        X = (

            data["x0"]

            +

            px * data["pixel_size"]

        )


        # Расчет координаты Y

        Y = (

            data["y0"]

            -

            py * data["pixel_size"]

        )


        return X, Y



    # ------------------


    # Основная функция локализации кадра камеры

    def process(self, frame):


        # Перевод изображения в оттенки серого

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # Поиск ключевых точек и дескрипторов кадра

        kp, desc = self.akaze.detectAndCompute(
            gray,
            None
        )


        # Если признаки не найдены

        if desc is None:

            return None



        # Получение соседних тайлов

        indexes = self.neighbors()


        tasks = []


        # Формирование задач для параллельного поиска

        for index in indexes:


            tasks.append(

                (

                    desc,

                    self.tiles[index]["desc"],

                    index

                )

            )



        # Поиск совпадений во всех соседних тайлах

        results = self.pool.map(
            match_tile,
            tasks
        )



        best = None



        # Выбор тайла с максимальным количеством совпадений

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



        # Если совпадений нет

        if best is None:

            return None



        # Проверка минимального количества совпадений

        if len(best["matches"]) < MIN_MATCHES:

            return None



        src = []

        dst = []



        # Формирование массивов координат совпавших точек

        for m in best["matches"]:


            src.append(

                kp[m[0]].pt

            )


            dst.append(

                best["tile"]["kp"]
                [m[1]]
                .pt

            )



        # Расчет матрицы преобразования между изображениями

        H, mask = cv2.findHomography(

            np.float32(src),

            np.float32(dst),

            cv2.RANSAC,

            5

        )



        # Проверка успешности вычисления преобразования

        if H is None:

            return None



        # Проверка количества корректных совпадений

        if np.sum(mask) < MIN_INLIERS:

            return None



        # Размер изображения камеры

        h, w = gray.shape


        # Координаты центра кадра камеры

        center = np.array(

            [

                [

                    [w/2, h/2]

                ]

            ],

            dtype=np.float32

        )



        # Перенос центра камеры в координаты тайла

        point = cv2.perspectiveTransform(

            center,

            H

        )



        px, py = point[0][0]



        # Обновление текущего тайла

        if best["index"] != self.current_index:


            print(

                "Переход:",

                self.current_index,

                "->",

                best["index"]

            )


            self.current_index = best["index"]



        # Перевод координат пикселя в мировую систему

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


    # Завершение работы процессов

    def close(self):


        self.pool.close()

        self.pool.join()