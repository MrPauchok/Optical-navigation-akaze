import os
import json
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
# =====================================================
# Настройки
# =====================================================
INPUT_TIFF = "map.tif"
OUTPUT_DIR = "tiles"
# Размер тайла:
# 3 x 1280
# 3 x 720
TILE_WIDTH = 3840
TILE_HEIGHT = 2160
# перекрытие тайлов
OVERLAP = 0.25
# сохранять дополнительные слои
SAVE_GRAY = True
# =====================================================
def save_pgw(filename, transform):
    """
    Сохранение геопривязки PNG
    """

    pgw = filename.replace(
        ".png",
        ".pgw"
    )

    with open(pgw, "w") as f:

        f.write(f"{transform.a}\n")
        f.write(f"{transform.b}\n")
        f.write(f"{transform.d}\n")
        f.write(f"{transform.e}\n")
        f.write(f"{transform.c}\n")
        f.write(f"{transform.f}\n")
def save_image(path, data):

    Image.fromarray(
        data
    ).save(
        path,
        compress_level=6
    )

def process_tile(rgb, base_path):
    save_image(
        base_path + ".png",
        rgb
    )
    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    if SAVE_GRAY:

        save_image(
            base_path + "_gray.png",
            gray
        )
def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    tiles = []
    with rasterio.open(INPUT_TIFF) as src:
        print()
        print(
            "Размер:",
            src.width,
            "x",
            src.height
        )
        print(
            "CRS:",
            src.crs
        )
        print(
            "Размер пикселя:",
            src.transform.a,
            "м"
        )
        if src.crs is None:
            raise Exception(
                "GeoTIFF не имеет CRS"
            )
        step_x = int(
            TILE_WIDTH *
            (1 - OVERLAP)
        )
        step_y = int(
            TILE_HEIGHT *
            (1 - OVERLAP)
        )
        positions = []
        for y in range(
            0,
            src.height - TILE_HEIGHT,
            step_y
        ):
            for x in range(
                0,
                src.width - TILE_WIDTH,
                step_x
            ):
                positions.append(
                    (x, y)
                )
        print(
            "Всего тайлов:",
            len(positions)
        )
        with tqdm(
            total=len(positions)
        ) as bar:
            for x, y in positions:


                window = Window(
                    x,
                    y,
                    TILE_WIDTH,
                    TILE_HEIGHT
                )
                transform = (
                    src.window_transform(
                        window
                    )
                )
                # координаты UTM
                coord_x = int(
                    transform.c
                )
                coord_y = int(
                    transform.f
                )
                name = (
                    f"{coord_x}_"
                    f"{coord_y}"
                )
                path = os.path.join(
                    OUTPUT_DIR,
                    name
                )
                data = src.read(
                    window=window
                )
                if data.shape[0] < 3:
                    continue


                rgb = data[:3]

                rgb = rgb.transpose(
                    1,
                    2,
                    0
                )
                process_tile(
                    rgb,
                    path
                )
                save_pgw(
                    path + ".png",
                    transform
                )
                tiles.append(
                    {
                        "file":
                            name + ".png",
                        "x":
                            coord_x,
                        "y":
                            coord_y,
                        "width":
                            TILE_WIDTH,
                        "height":
                            TILE_HEIGHT,
                        "pixel_size":
                            src.transform.a
                    }
                )


                bar.update(1)



    with open(
        os.path.join(
            OUTPUT_DIR,
            "index.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            tiles,
            f,
            indent=4,
            ensure_ascii=False
        )


    print()
    print(
        "Готово."
    )



if __name__ == "__main__":
    main()