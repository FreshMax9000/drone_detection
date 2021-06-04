import logging
import time
import math
from typing import List
from itertools import combinations

import numpy as np
from numpy.lib import average

from .single_auswertung import Spot, BildAuswertung


class DetectionMaker:

    def __init__(self, distance_threshold: float):
        self.distance_threshold = distance_threshold
        logging.debug(f"{self.__class__} initialisiert")

    def _get_ebene(self, drone_vec: np.ndarray):
        """Berechnet zwei zu sich gegenseitig und zu drone_vec orthogonale Vektoren"""
        random_vec = np.array([1, 1, 1])
        vector_1 = np.cross(drone_vec, random_vec)
        vector_2 = np.cross(drone_vec, vector_1)
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        vector_2 = vector_2 / np.linalg.norm(vector_2)
        return (vector_1, vector_2)

    def make_detection1(self):
        """Bestimmt die Lokation von möglichen Drohnen anhand von den von BildAuswertung gegebenen Daten"""
        detection_list = []
        spotted_list, spot_time = self.auswertung.get_spotted()
        # Prüfe jeden gegeben Spot auf Kollision mit jedem anderen Spot
        for i, spot_1 in enumerate(spotted_list):
            plane_vec_1, plane_vec_2 = self._get_ebene(spot_1.drone_vec)
            trans_2d_to_3d = np.array([plane_vec_1, plane_vec_2])
            trans_3d_to_2d = np.linalg.pinv(trans_2d_to_3d)
            r = spot_1.radius
            c = np.dot(spot_1.base_pos, trans_3d_to_2d)[0] # c: x position of middle of circle in 2d space
            d = np.dot(spot_1.base_pos, trans_3d_to_2d)[1] # d: y position of middle of circle in 2d space
            for spot_2 in spotted_list[i+1:]:
                # project spot 2 on to the plane of spot_1.drone_vec
                spot_2_base_flat = np.dot(spot_2.base_pos, trans_3d_to_2d)
                spot_2_dir_flat = np.dot(spot_2.drone_vec, trans_3d_to_2d)
                # calculate linear function parameters based on spot_2
                if spot_2_dir_flat[0] == 0.0: # Check if a should be infinity or nan
                    if spot_2_dir_flat[1] == 0.0: # a would be nan
                        print("abbruch weil parallel")
                        continue # spot_2_dir_is orthogonal to spot_1_dir plane;
                                 # therefore there should be interceptions
                    else:
                        print("a ist unendlich")
                        a = 1e10 # if a should be infity make it rather big
                else:
                    a = spot_2_dir_flat[1] / spot_2_dir_flat[0] # calculate a properly
                b = spot_2_base_flat[1] - a * spot_2_base_flat[0]
                b_delta = np.sqrt((spot_2.radius**2)+((a*spot_2.radius)**2)) # todo schönere funktion für euklidsche distanz
                # Calculate parallel functions to linear function of spot_2
                # representing the outer lines of the cylinder
                b_r_plus = b + b_delta
                b_r_minus = b - b_delta
                # Calculate intersection between circle of spot_1 and linear functions of spot_2
                intersection_x_1 = np.roots([1+a**2, 2*a*b_r_plus-2*a*d-2*c, b_r_plus**2+c**2+d**2-r**2-2*b*d])
                intersection_x_2 = np.roots([1+a**2, 2*a*b_r_minus-2*a*d-2*c, b_r_minus**2+c**2+d**2-r**2-2*b*d])
                for intersection in [intersection_x_1, intersection_x_2]:
                    if any(np.isreal(intersection)):
                        #print(intersection)
                        x = sum(intersection) / len(intersection)
                        intersection_2d_pos = np.array([x, a*x + b])
                        intersection_3d_pos = np.dot(intersection_2d_pos, trans_2d_to_3d)
                        #print(intersection_3d_pos)
                        detection_list.append(
                            (spot_1.camera_name, spot_2.camera_name, intersection_3d_pos)
                        ) # todo fix double entry bug
        return detection_list

    def get_3d_to_2d_trans(self, spot_1: Spot, spot_2: Spot):
        return np.array([spot_1.drone_vec, spot_2.drone_vec])

    def merge_detections(self, detection_list: List[np.ndarray]) -> List[np.ndarray]:
        #for detec1, detec2 in combinations(detection_list, 2):

        #print(detection_list)
        if len(detection_list) < 2:
            return detection_list
        filtered_detections = self.merge_detections(detection_list[1:])
        similiar_index = []
        for i, detection in enumerate(filtered_detections):
            if np.linalg.norm(detection_list[0] - detection) < self.distance_threshold:
                similiar_index.append(i)
        similiar_values = [detec for i, detec in enumerate(filtered_detections) if i in similiar_index]
        similiar_values.append(detection_list[0])
        detection_list[0] = np.mean(similiar_values, axis=0)
        #detections = [
        #    detection for detection in filtered_detections if (detection not in similiar_values)
        #    ]
        detections = [detec for i, detec in enumerate(filtered_detections) if i not in similiar_index]
        detections.append(detection_list[0])
        return detections

    def make_detection2(self, spot_list: List[Spot], spot_time: float) -> List[np.ndarray]:
        """Bestimmt die Lokation von möglichen Drohnen anhand von den von BildAuswertung gegebenen Daten"""
        detection_list = []
        for i, spot_1 in enumerate(spot_list):
            for spot_2 in spot_list[i+1:]:
                # Bestimme Ebene (Also Transformation)
                #   Ebene geht durch Ursprung und wird durch den drone_vec der beiden zu prüfenden
                #   Spots aufgespannt
                trans_3d_to_2d = self.get_3d_to_2d_trans(spot_1, spot_2)
                trans_2d_to_3d = np.linalg.pinv(trans_3d_to_2d)
                # Übertrage Spots in die Ebene
                #   pos_vec und drone_vec einzeln
                spot_1_2d = spot_1.get_2d(trans_3d_to_2d)
                spot_2_2d = spot_2.get_2d(trans_3d_to_2d)
                # Geradengleichung in der Ebene für beide Spots aufstellen
                # Geradengleichung gleichstellen und lösen -> s_1, s_2
                b = spot_2_2d.base_pos - spot_1_2d.base_pos
                a = np.array([spot_1_2d.raw_drone_vec, -spot_2_2d.raw_drone_vec]).T
                try:
                    s_1, s_2 = np.linalg.solve(a, b)
                except np.linalg.LinAlgError:
                    logging.warning("Parallel spottings")
                    continue
                # Mit s_1 und s_2 dreidimsensionale "Schnittpunkte" der Spots berechnen
                #p_1 = np.dot(trans_2d_to_3d, spot_1_2d.get_point(s_1))
                #p_2 = np.dot(trans_2d_to_3d, spot_2_2d.get_point(s_2))
                p_1 = spot_1.get_point(s_1) # todo so richtig ??? -> weiter testen !
                p_2 = spot_2.get_point(s_2)
                #print(p_1)
                #print(p_2)
                # Distanz zwischen den Schnittpunkten berechnen
                # Wenn Distanz unter bestimmten Schwellwert, Mittelpunkt zwischen Schnittpunkten in Liste eintragen
                if np.linalg.norm(p_1 - p_2) <= self.distance_threshold:
                    #print(f"Spotted between {spot_1.camera_name} and {spot_2.camera_name}")
                    detection_list.append((p_1 + p_2) / 2)
        # Für Alle Schnittpunkte mit jedem Schnittpunkt
        #   Wenn anderer Schnittpunkt sehr nahe an eigenem Schnittpunkt:
        #       Zusammenführen
        detection_list = self.merge_detections(detection_list)
        return detection_list

    def make_detection(self, spot_list: List[Spot], spot_time: float) -> List[np.ndarray]:
        return self.make_detection2(spot_list, spot_time)



if __name__ == "__main__":
    diesdas = DetectionMaker()
    #print(diesdas.make_detection())
    for detection in diesdas.make_detection2():
        print(detection)
    print("Execution finished")