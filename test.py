import time

import cv2
import numpy as np

from src.image_data import ImageData
from src.single_auswertung import BildAuswertung, Spot
from src.detection import DetectionMaker


def get_spotted_dummy():
    e1 = Spot(
        "cam 0",
        np.array([0, 0, 0]), # Base pos
        np.array([0, 0, 1]), # Drone vector
        1, # size
        #time.time() - 0.2 # single time
        1
    )
    e2 = Spot(
        "cam 1",
        np.array([1, 0, 0]), # Base pos
        np.array([-0.2, 0, 1]), # Drone vector
        1, # size
        #time.time() - 0.2 # single time
        1
    )
    e3 = Spot(
        "cam 2",
        np.array([0, 1, 0]), # Base pos
        np.array([0.0, -0.2, 1]), # Drone vector
        1, # size
        #time.time() - 0.2 # single time
        1
    )
    return ([e1, e2, e3], time.time())


def test_detection():
    d_spot_list, d_timestamp = get_spotted_dummy()
    detector = DetectionMaker(0.5)
    print(detector.make_detection2(d_spot_list, d_timestamp))


def main():
    test_detection()
    return

    dummy_img = cv2.imread("/home/max/darknet_test/darknet/data/eagle.jpg")
    dummy_img_data = ImageData(dummy_img, time.time(), "some_esp", np.array([0, 0, 0]))
    #print(np.shape(dummy_img_data.image))
    cv2.imshow("dingledangle", dummy_img_data.image)
    print(cv2.waitKey(10000))
    #return
    auswertung = BildAuswertung()
    spotted = auswertung.get_spotted({"some_esp": dummy_img_data})
    print(spotted[0])

if __name__ == "__main__":
    main()