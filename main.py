from capture import ScreenCapture
from localization import Localizer

from openpyxl import Workbook

import time


FEATURES_DIR = "features"


def main():

    capture = ScreenCapture()

    localizer = Localizer(
        FEATURES_DIR
    )

    wb = Workbook()

    ws = wb.active

    ws.title = "Trajectory"

    ws.append(
        [
            "Time",
            "Tile",
            "X",
            "Y"
        ]
    )

    print(
        "Система запущена"
    )

    last = time.time()

    try:

        while True:

            frame = capture.read()

            result = localizer.process(
                frame
            )

            if result:

                tile, x, y = result

                print(
                    f"{tile} | "
                    f"X={x:.2f} "
                    f"Y={y:.2f}"
                )

                ws.append(
                    [
                        time.time(),
                        tile,
                        float(x),
                        float(y)
                    ]
                )

            else:

                print(
                    "Нет решения"
                )

            now = time.time()

            fps = 1 / (now - last)

            last = now

            print(
                "FPS:",
                round(fps, 2)
            )

            print(
                "-" * 40
            )

    except KeyboardInterrupt:

        print(
            "Остановка"
        )

    finally:

        wb.save(
            "coordinates.xlsx"
        )

        localizer.close()


if __name__ == "__main__":

    main()