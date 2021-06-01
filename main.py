import time

import cv2
import numpy as np

from src.single_erfassung import ThreadController
from src.single_auswertung import BildAuswertung


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
        np.array([0, 0.07, 0])
    ]

    auswertung = BildAuswertung()

    with ThreadController(ESP_32_URL_LIST, ESP_32_POS_LIST, esp32_name_list=ESP_32_NAME_LIST) as img_threads:
        start_time = time.time()
        while time.time() - start_time < 120:
            tmp_time = time.time()
            image_dict = img_threads.get_image_dict()
            tmp_time = time.time() - tmp_time
            print(f"Getting picture dict took {tmp_time}s")
            tmp_time = time.time()
            for raw_spot in auswertung.get_spotted(image_dict):
                print(raw_spot)
            tmp_time = time.time() - tmp_time
            print(f"Analying and printing the detections took {tmp_time}s")
            # todo further processing



    print("--- Execution finished ---")


if __name__ == "__main__":
    main()
