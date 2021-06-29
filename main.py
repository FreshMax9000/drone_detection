import time
from statistics import mean

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from src.single_erfassung import ThreadController
from src.single_auswertung import BildAuswertung
from src.detection import DetectionMaker


DISTANCE_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.235


def show_imgs(image_dict: dict, spot_list: list, sleep_time, timestamp = time.time()) -> None:
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
    #cv2.imwrite(f"./orig_test_2/{timestamp: .1f}.png", h)
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

    auswertung = BildAuswertung(DETECTION_THRESHOLD, use_scissors=False)
    detector = DetectionMaker(DISTANCE_THRESHOLD)

    with ThreadController(ESP_32_URL_LIST, ESP_32_POS_LIST, esp32_name_list=ESP_32_NAME_LIST) as img_threads:
        print("Detection running !")
        start_time = time.time()
        x_list = []
        y_list = []
        z_list = []
        abc_time = [[],[],[]]
        time.sleep(1)
        while time.time() - start_time < 30:
            tmp_time = time.time()
            image_dict, timestamp = img_threads.get_image_dict()
            tmp_time = time.time() - tmp_time
            print(f"Getting picture dict took {tmp_time}s")
            abc_time[0].append(tmp_time)
            tmp_time = time.time()
            spot_list, image_dict = auswertung.get_spotted(image_dict)
            #print(len(spot_list))
            #[print(spot) for spot in spot_list]
            tmp_time = time.time() - tmp_time
            print(f"Analyzing single picture took {tmp_time}s")
            abc_time[1].append(tmp_time)
            tmp_time = time.time()
            detections = detector.make_detection(spot_list, timestamp)
            tmp_time = time.time() - tmp_time
            print(f"Making detections in 3d took {tmp_time}s")
            abc_time[2].append(tmp_time)
            n_timestamp = time.time()-start_time
            #print(f"{n_timestamp:.1f}: {detections}")
            show_imgs(image_dict, spot_list, 1, timestamp=n_timestamp)
            for detection in detections:
                x_list.append((n_timestamp, detection[0]))
                y_list.append((n_timestamp, detection[1]))
                z_list.append((n_timestamp, detection[2]))
        #plt.plot(*zip(*x_list), "r.", label="x pos")
        #plt.plot(*zip(*y_list), "b.", label="y pos")
        #plt.plot(*zip(*z_list), "g.", label="z pos")
        #plt.xlabel("time[s]")
        #plt.ylabel("detected position[m]")
        #plt.legend()
        #plt.savefig("pos_graph.png")
        a_time = mean(abc_time[0])
        b_time = mean(abc_time[1])
        c_time = mean(abc_time[2])
        print(f"Retrieval: {a_time}s")
        print(f"Detection: {b_time}s")
        print(f"Position: {c_time}s")
        print(f"Total: {a_time+b_time+c_time}")
        plt.bar([0, 1, 2], [a_time, b_time, c_time], tick_label=["Bilderfassung", "Objekterkennung", "3D Lokalisierung"], log=True)
        plt.ylabel("time spent during step[s]")
        plt.savefig("time_graph.png")



    print("--- Execution finished ---")


if __name__ == "__main__":
    main()
