import time

import cv2
import numpy as np

from src.single_erfassung import ThreadController
from src.single_auswertung import BildAuswertung
from src.detection import DetectionMaker


DISTANCE_THRESHOLD = 0.5


def main():
    ESP_32_URL_LIST = [
        "http://192.168.188.80/capture",
        "http://192.168.188.82/capture",
    ]
    ESP_32_NAME_LIST = [
        "top_esp",
        "bottom_esp"
    ]
    ESP_32_POS_LIST = [
        np.array([0, 0, 0]),
        np.array([0, 0.06, 0])
    ]

    auswertung = BildAuswertung()
    detector = DetectionMaker(DISTANCE_THRESHOLD)

    with ThreadController(ESP_32_URL_LIST, ESP_32_POS_LIST, esp32_name_list=ESP_32_NAME_LIST) as img_threads:
        print("Detection running !")
        start_time = time.time()
        while time.time() - start_time < 300:
            tmp_time = time.time()
            image_dict, timestamp = img_threads.get_image_dict()
            tmp_time = time.time() - tmp_time
            #print(f"Getting picture dict took {tmp_time}s")
            tmp_time = time.time()
            spot_list = auswertung.get_spotted(image_dict)
            tmp_time = time.time() - tmp_time
            #print(f"Analying and printing the detections took {tmp_time}s")
            print(f"{time.time():.0f}: {detector.make_detection(spot_list, timestamp)}")



    print("--- Execution finished ---")


if __name__ == "__main__":
    main()
