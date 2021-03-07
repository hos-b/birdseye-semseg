import numpy as np
import torch

__carla_to_rgb_palette = {
    0 : [0, 0, 0],          # Unlabeled [filtered]
    1 : [70, 70, 70],       # Building
    2 : [100, 40, 40],      # Fence [filtered]
    3 : [55, 90, 80],       # Other
    4 : [220,  20,  60],    # Pedestrian [not filtered but not available either]
    5 : [153, 153, 153],    # Pole [filtered]
    6 : [157, 234, 50],     # RoadLine
    7 : [128, 64, 128],     # Road
    8 : [244, 35, 232],     # SideWalk
    9 : [107, 142, 35],     # Vegetation
    10 : [0, 0, 142],       # Vehicles
    11 : [102, 102, 156],   # Wall
    12 : [220, 220, 0],     # TrafficSign
    13 : [70, 130, 180],    # Sky [filtered]
    14 : [81, 0, 81],       # Ground
    15 : [150, 100, 100],   # Bridge
    16 : [230, 150, 140],   # RailTrack
    17 : [180, 165, 180],   # GuardRail [filtered]
    18 : [250, 170, 30],    # TrafficLight [filtered]
    19 : [110, 190, 160],   # Static
    20 : [170, 120, 50],    # Dynamic
    21 : [45, 60, 150],     # Water
    22 : [145, 170, 100]    # Terrain
}

__ours_to_rgb_palette = {
    0: [70, 70, 70],
    1: [55, 90, 80],
    2: [128, 64, 128],
    3: [244, 35, 232],
    4: [107, 142, 35],
    5: [0, 0, 142],
    6: [45, 60, 150]
}
__our_classes = {
    0: 'Buildings',
    1: 'Misc',
    2: 'Road',
    3: 'SideWalk',
    4: 'Vegetation',
    5: 'Vehicles',
    6: 'Water'
}
__carla_to_our_ids = {
    # 0, 2, 4, 5, 12, 13 & 17 are assumed to be absent in the dataset
    1 : 0,  # Buildings   -> Buildings
    3 : 1,  # Other       -> Misc
    6 : 2,  # RoadLine    -> Road (for now)
    7 : 2,  # Road        -> Road
    8 : 3,  # SideWalk    -> SideWalk
    9 : 4,  # Vegetation  -> Vegetation
    10 : 5, # Vehicles    -> Vehicles
    11 : 0, # Walls       -> Buildings
    12 : 1, # TrafficSign -> Misc
    14 : 3, # Ground      -> SideWalk
    15 : 0, # Bridge      -> Buildings
    16 : 1, # RailTrack   -> Misc
    19 : 1, # Static      -> Misc
    20 : 1, # Dynamic     -> Misc
    21 : 6, # Water       -> Water
    22 : 4  # Terrain     -> Vegetation
}

def carla_semantics_to_cityscapes_rgb(semantic_ids : torch.Tensor) -> np.ndarray:
    assert len(semantic_ids.shape) == 2, f'expected HxW, got {semantic_ids.shape}'
    semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                     semantic_ids.shape[1], 3),
                              dtype=np.uint8)
    for sid, cityscapes_rgb in __carla_to_rgb_palette.items():
        semantic_rgb[semantic_ids == sid] = cityscapes_rgb
    return semantic_rgb

def carla_semantics_to_our_semantics(semantic_ids : np.ndarray) -> np.ndarray:
    assert len(semantic_ids.shape) == 2, f'expected HxW, got {semantic_ids.shape}'
    our_semantics = np.zeros_like(semantic_ids)
    for carla_id, our_id in __carla_to_our_ids.items():
        our_semantics[semantic_ids == carla_id] = our_id
    return our_semantics

def our_semantics_to_cityscapes_rgb(semantic_ids : torch.Tensor) -> np.ndarray:
    assert len(semantic_ids.shape) == 2, f'expected HxW, got {semantic_ids.shape}'
    # not really cityscapes but cityscapes-like
    semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                     semantic_ids.shape[1], 3),
                              dtype=np.uint8)
    for sid, cityscapes_rgb in __ours_to_rgb_palette.items():
        semantic_rgb[semantic_ids == sid] = cityscapes_rgb
    return semantic_rgb