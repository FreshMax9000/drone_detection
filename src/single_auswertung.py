import time
from threading import Thread

import numpy as np

from .darknet import Net
from .image_data import ImageData


class RawSpot:

    def __init__(self, camera_name: str, base_pos: np.array, single_time: time.time) -> None:
        self.camera_name = camera_name
        self.base_pos = base_pos
        self.single_time = single_time
        self.spotted_list = []

    def __str__(self):
        return f"{self.camera_name}: {self.spotted_list}"


class Spot:

    def __init__(self, camera_name, base_pos, drone_vec, size, single_time) -> None:
        self.camera_name = camera_name
        self.base_pos = base_pos
        self.drone_vec = drone_vec / np.linalg.norm(drone_vec)
        self.size = size
        self.single_time = single_time

    def __str__(self):
        return f"bPos: {self.base_pos}; dVec: {self.drone_vec}"

    def copy(self, base_pos=None, drone_vec=None):
        if base_pos is None or drone_vec is None:
            base_pos = self.base_pos
            drone_vec = self.drone_vec
        return Spot(self.camera_name, base_pos, drone_vec, self.size, self.single_time)

    def get_2d(self, trans: np.array):
        base_pos = np.dot(trans, self.base_pos)
        drone_vec = np.dot(trans, self.drone_vec)
        return self.copy(base_pos=base_pos, drone_vec=drone_vec)

    def get_point(self, s):
        return self.base_pos + s * self.drone_vec

    @property
    def radius(self):
        return self.size / 2.0


class BildAuswertung:
    """Class to run YOLO on single images and extract possible drone spottings from them.
    """

    def __init__(self):
        """Initializes a BildAuswertung object.

        This usually includes initializing a Net (darknet) object with yolo data, which takes a while (~2s).
        """
        self.darknet = Net("/home/max/darknet_test/darknet/libdarknet.so",
                           "/home/max/darknet_test/darknet/yolov3.weights",
                           "/home/max/darknet_test/darknet/cfg/yolov3.cfg",
                           "/home/max/darknet_test/darknet/cfg/coco.data")

    def fill_raw_spot(self, raw_spot: RawSpot, img) -> None:
        raw_spot.spotted_list.extend(self.darknet.detect(img))

    def get_spotted(self, image_dict: dict) -> list:
        """Generates a list of Spot objects from the given image_dict.

        All Images in image_dict are searched for drone objects. Each found drone object is
        described by a Spot object. A list containing all found Spots is then returned

        Args:
            image_dict (dict): A dict containing ImageData objects

        Returns:
            list: A list containing Spot objects, which correspond the found drones in
                the given images.
        """
        raw_spot_list = []
        for key in image_dict:
            raw_spot = RawSpot(image_dict[key].esp_name, image_dict[key].pos, image_dict[key].timestamp)
            self.fill_raw_spot(raw_spot, image_dict[key].image)
            raw_spot_list.append(raw_spot)
        return raw_spot_list
        # todo Generate Spot from rawSpot using the maximum width of the orig img and stuff

    #def get_spotted(self):
    #    return self.get_spotted_dummy()

    def get_spotted_dummy(self):
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
            np.array([0, 1, 1]), # Drone vector
            1, # size
            #time.time() - 0.2 # single time
            1
        )
        e3 = Spot(
            "cam 2",
            np.array([0, 1, 0]), # Base pos
            np.array([0, -0.7, 0.7]), # Drone vector
            1, # size
            #time.time() - 0.2 # single time
            1
        )
        return ([e1, e2], time.time())
