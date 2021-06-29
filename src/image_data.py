
import cv2
import numpy as np


class ImageData:
    """Class containing an image and metadata about it.
    """

    folder_path = "/home/max/Pictures/drone_demo_pics"

    def __init__(self, image, timestamp: float, esp_name: str, pos: np.ndarray) -> None:
        self.image = image
        self.timestamp = timestamp # Approximate time it was taken
        self.esp_name = esp_name # Name of the microcontroller which took the image
        self.pos = pos # 3d position of microcontroller which took the image

    def save_to_file(self):
        cv2.imwrite(self.folder_path+f"/{self.timestamp}-{self.esp_name}.jpg", self.image)
