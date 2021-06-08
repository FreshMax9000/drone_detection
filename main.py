import time

import cv2
import numpy as np

from src.single_erfassung import ThreadController
from src.single_auswertung import BildAuswertung
from src.detection import DetectionMaker


DISTANCE_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.3


def show_imgs(image_dict: dict, spot_list: list, sleep_time) -> None:
    if len(image_dict) != 4:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key, image_data in image_dict.items():
        cv2.putText(image_data.image, image_data.esp_name, (10, 460), font, 3, (0, 255, 0), thickness=2)
    keys = list(image_dict.keys())
    v1 = np.vstack((image_dict["esp_1"].image, image_dict["esp_0"].image))
    v2 = np.vstack((image_dict["esp_3"].image, image_dict["esp_2"].image))
    h = np.hstack((v1, v2))
    cv2.imshow("live feed", h)
    cv2.waitKey(sleep_time)
 

def main():
    ESP_32_URL_LIST = [
        "http://192.168.188.74/capture",
        "http://192.168.188.80/capture",
        "http://192.168.188.84/capture",
        "http://192.168.188.82/capture",
    ]
    ESP_32_NAME_LIST = [
        "esp_0",
        "esp_1",
        "esp_2",
        "esp_3"
    ]
    ESP_32_POS_LIST = [
        np.array([-0.125, -0.125, 0]),
        np.array([-0.125, 0.125, 0]),
        np.array([0.125, -0.125, 0]),
        np.array([0.125, 0.125, 0])
    ]

    auswertung = BildAuswertung(DETECTION_THRESHOLD, use_scissors=True)
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
            spot_list, image_dict = auswertung.get_spotted(image_dict)
            #print(len(spot_list))
            #[print(spot) for spot in spot_list]
            tmp_time = time.time() - tmp_time
            #print(f"Analying and printing the detections took {tmp_time}s")
            print(f"{time.time()-start_time:.1f}: {detector.make_detection(spot_list, timestamp)}")
            show_imgs(image_dict, spot_list, 1)



    print("--- Execution finished ---")


if __name__ == "__main__":
    main()
