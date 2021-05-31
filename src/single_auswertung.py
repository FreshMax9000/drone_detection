import time

import numpy as np


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

    def __init__(self):
        pass

    def get_spotted(self):
        return self.get_spotted_dummy()

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
