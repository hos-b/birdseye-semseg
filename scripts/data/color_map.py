import numpy as np


def semantic_to_cityscapes(semantic_ids : np.ndarray):
    semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                     semantic_ids.shape[1], 3), 
                              dtype=np.uint8)
    semantic_rgb[semantic_ids == 0] = [0, 0, 0]         # Unlabeled
    semantic_rgb[semantic_ids == 1] = [70, 70, 70]      # Building
    semantic_rgb[semantic_ids == 2] = [100, 40, 40]     # Fence
    semantic_rgb[semantic_ids == 3] = [55, 90, 80]      # Other
    semantic_rgb[semantic_ids == 4] = [220,  20,  60]   # Pedestrian
    semantic_rgb[semantic_ids == 5] = [153, 153, 153]   # Pole
    semantic_rgb[semantic_ids == 6] = [157, 234, 50]    # RoadLine
    semantic_rgb[semantic_ids == 7] = [128, 64, 128]    # Road
    semantic_rgb[semantic_ids == 8] = [244, 35, 232]    # SideWalk
    semantic_rgb[semantic_ids == 9] = [107, 142, 35]    # Vegetation
    semantic_rgb[semantic_ids == 10] = [0, 0, 142]      # Vehicles
    semantic_rgb[semantic_ids == 11] = [102, 102, 156]  # Wall
    semantic_rgb[semantic_ids == 12] = [220, 220, 0]    # TrafficSign
    semantic_rgb[semantic_ids == 13] = [70, 130, 180]   # Sky
    semantic_rgb[semantic_ids == 14] = [81, 0, 81]      # Ground
    semantic_rgb[semantic_ids == 15] = [150, 100, 100]  # Bridge
    semantic_rgb[semantic_ids == 16] = [230, 150, 140]  # RailTrack
    semantic_rgb[semantic_ids == 17] = [180, 165, 180]  # GuardRail
    semantic_rgb[semantic_ids == 18] = [250, 170, 30]   # TrafficLight
    semantic_rgb[semantic_ids == 19] = [110, 190, 160]  # Static
    semantic_rgb[semantic_ids == 20] = [170, 120, 50]   # Dynamic
    semantic_rgb[semantic_ids == 21] = [45, 60, 150]    # Water
    semantic_rgb[semantic_ids == 22] = [145, 170, 100]  # Terrain
    return semantic_rgb