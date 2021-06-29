"""This module contains the first stage of the detection pipeline.

The first stage contains the ESP32-Microcontrollers and a polling mechanism delivering
the data from the microcontrollers to the rest of the software."""

import threading
import time
from typing import Callable, Iterable, Mapping, Tuple, Dict
import copy

import cv2
import numpy as np
from skimage import io

from .image_data import ImageData


class GetImageThread(threading.Thread):
   """Thread to poll images from a single microcontroller.
   
   For each microcontroller one instance of this class should exist. This class is then
   permanently polling images from the microcontroller. The images are stored in the shared
   variable image_dict, to which access is controlled by image_dict_lock. All threads store their
   most current images in this dictionary, which then can also be accessed by other parts of the
   software. For accessing the shared variable see class "ThreadController".
   """

   image_dict: dict = {}
   image_dict_lock: threading.Lock = threading.Lock()
   exit = False

   def __init__(
      self,
      esp_url: str,
      esp_name: str,
      pos: np.array,
      group: None = None,
      target: Callable = None,
      name: str = None,
      args: Iterable = (),
      kwargs: Mapping = {},
      *,
      daemon: bool = None
      ) -> None:
      """Intialize GetImageThread object.

      For further documentation on the arguments supplied to the threading.Thread parent class
      see https://docs.python.org/3/library/threading.html#thread-objects.

      Args:
          esp_url (str): URL to ESP, eg 'http://192.168.188.80/capture'
          esp_name (str): Name of the ESP this Thread is responsible for
          pos (np.array): 3d position of this threads esp
          group (None, optional): Defaults to None.
          target (Callable, optional): Defaults to None.
          name (str, optional): Defaults to None.
          args (Iterable, optional): Defaults to ().
          kwargs (Mapping, optional): Defaults to {}.
          daemon (bool, optional): Defaults to None.
      """
      if name is None:
         name = f"{esp_name}-Thread"
      super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
      self.esp_url = esp_url
      self.esp_name = esp_name
      self.pos = pos

   def _get_frame(self):
      img = io.imread(self.esp_url) # Acquire image via http
      img = cv2.flip(img, 0) # Vertically flip the image (top->bottom and bottom->top)
      return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Set correct color

   def run(self):
      while not self.exit:
         img = self._get_frame() # get single image
         img_data = ImageData(img, time.time(), self.esp_name, self.pos)
         # Acquire lock and store image in shared variable
         self.image_dict_lock.acquire(blocking=True)
         self.image_dict.update({self.esp_name: img_data})
         self.image_dict_lock.release()


class ThreadController:
   """Class to control threads getting pictures from the esp32s.
   """

   def __init__(self, esp32_url_list: list, esp32_pos_list: list, esp32_name_list: list = None):
      """Initialize ThreadController objekct

      Args:
          esp32_url_list (list): List of URLs to access the image data of the esp32s
          esp32_pos_list (list): List of positions of the esp32s. Preferably in the same
            order as esp32_url_list
          esp32_name_list (list, optional): List of names for the esp32s. Preferably in the same
            order as esp32_url_list. Defaults to None.
      """
      self.threads = []
      # Generate a GetImageThread object for each microcontroller
      for i, url in enumerate(esp32_url_list):
         if esp32_name_list is None:
            self.threads.append(GetImageThread(url, f"ESP{i}", esp32_pos_list[i]))
         else:
            self.threads.append(GetImageThread(url, esp32_name_list[i], esp32_pos_list[i]))

   def start_threads(self):
      for thread in self.threads:
         thread.start()

   def get_image_dict(self) -> Tuple[Dict[str, ImageData], float]:
      """Retrievies the most up-to-date pictures of the esp32s and returns them in a dict with a timestamp.

      Returns:
          tuple:
            dict: A dict mapping the names of the esp32s to ImageData objects. Usually looks somewhat like this:
            {"esp1": image_data_obj_1,
            "esp": image_data_obj_2}
         float: Timestamp when the images have been acquired from the ESPs
      """
      # Acquire lock and get image from shared variable
      GetImageThread.image_dict_lock.acquire(blocking=True)
      glob_time = time.time()
      image_dict = copy.copy(GetImageThread.image_dict)
      GetImageThread.image_dict_lock.release()
      return image_dict, glob_time

   def stop_all_threads(self):
      GetImageThread.exit = True

   def __enter__(self):
      self.start_threads()
      return self

   def __exit__(self, exception_type, exception_value, exception_traceback):
      self.stop_all_threads()


if __name__ == "__main__":
   url_list = ["http://10.42.0.124/capture", "http://10.42.0.206/capture"]
   name_list = ["uff", "kek"]
   controller = ThreadController(url_list, name_list)
   with controller as c:     
      for i in range(5):
         images = c.get_image_dict()
         for key in images:
            images[key].save_to_file()
         time.sleep(1)
