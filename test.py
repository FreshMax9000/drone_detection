import time

import cv2
import numpy as np

from src.image_data import ImageData
from src.single_auswertung import BildAuswertung, Spot
from src.detection import DetectionMaker
from src.single_erfassung import ThreadController


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

def dummdumm():
    some_list = ['a', 'b', 'c', 'd']
    print(some_list[1:])


def get_demo_pics():
    url_list = ["http://192.168.188.80/capture",
                "http://192.168.188.82/capture"]
    name_list = ["uff", "kek"]
    controller = ThreadController(url_list, [np.array([0, 0, 0]), np.array([0, 0.06, 0])], esp32_name_list=name_list)
    with controller as c:     
        for i in range(2):
            images = c.get_image_dict()[0]
            
            for key in images:
                images[key].save_to_file()
                break
            time.sleep(2)


def test_bildauswertung():
    auswertung = BildAuswertung(0.235, use_scissors=False)
    img = cv2.imread("/home/max/Documents/drone_detection/test_images/drone_1.jpg")
    #img = cv2.imread("/home/max/darknet_test/darknet/data/dog.jpg")
    dings = ImageData(img, 0, "esp_test", np.array([0.0, 0.0, 0.0]))
    spot_list, image_dict = auswertung.get_spotted({"esp_test": dings})
    cv2.imshow("test", image_dict["esp_test"].image)
    print(spot_list[0])
    cv2.waitKey(0)



def main():
    #dummdumm()
    #return
    #test_detection()
    #get_demo_pics()
    test_bildauswertung()
    return

    dummy_img = cv2.imread("/home/max/darknet_test/darknet/data/eagle.jpg")
    dummy_img_data = ImageData(dummy_img, time.time(), "some_esp", np.array([0, 0, 0]))
    #print(np.shape(dummy_img_data.image))
    cv2.imshow("dingledangle", dummy_img_data.image)
    print(cv2.waitKey(10000))
    #return
    auswertung = BildAuswertung("yolov5m.pt")
    spotted = auswertung.get_spotted({"some_esp": dummy_img_data})
    #print(spotted[0])

if __name__ == "__main__":
    main()