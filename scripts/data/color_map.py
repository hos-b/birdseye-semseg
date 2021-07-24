import numpy as np
import torch


# Semantic IDs to RGB -------------------------------------------------------------
__carla_to_rgb_palette = {
    0  : [0, 0, 0],          # Unlabeled [filtered]
    1  : [70, 70, 70],       # Building
    2  : [100, 40, 40],      # Fence [filtered]
    3  : [55, 90, 80],       # Other
    4  : [220,  20,  60],    # Pedestrian [not filtered but not available either]
    5  : [153, 153, 153],    # Pole [filtered]
    6  : [157, 234, 50],     # RoadLine
    7  : [128, 64, 128],     # Road
    8  : [244, 35, 232],     # SideWalk
    9  : [107, 142, 35],     # Vegetation
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
__oursplusmask_to_rgb_palette = {
    0: [70, 70, 70],
    1: [55, 90, 80],
    2: [128, 64, 128],
    3: [244, 35, 232],
    4: [107, 142, 35],
    5: [0, 0, 142],
    6: [45, 60, 150],
    7: [0, 0, 0] # mask is black
}
__diminished_to_rgb_palette = {
    0: [0, 0, 142],
    1: [128, 64, 128],
    2: [244, 35, 232]
}
# Semantic IDs to Class String ----------------------------------------------------
__carla_classes = {
    0 : 'Unlabeled',
    1 : 'Building',
    2 : 'Fence',
    3 : 'Other',
    4 : 'Pedestrian',
    5 : 'Pole',
    6 : 'RoadLine',
    7 : 'Road',
    8 : 'SideWalk',
    9 : 'Vegetation',
    10 : 'Vehicles',
    11 : 'Wall',
    12 : 'TrafficSign',
    13 : 'Sky',
    14 : 'Ground',
    15 : 'Bridge',
    16 : 'RailTrack',
    17 : 'GuardRail',
    18 : 'TrafficLight',
    19 : 'Static',
    20 : 'Dynamic',
    21 : 'Water',
    22 : 'Terrain'
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
__our_classes_plus_mask = {
    0: 'Buildings',
    1: 'Misc',
    2: 'Road',
    3: 'SideWalk',
    4: 'Vegetation',
    5: 'Vehicles',
    6: 'Water',
    7: 'Mask'
}
__diminished_classes = {
    0: 'Vehicles',
    1: 'Drivable',
    2: 'Non-drivable'
}
# CARLA Semantic IDs to Subset IDs ------------------------------------------------
__carla_to_our_ids = {
    # 0, 4, 5, 12, 13 are assumed to be absent in the dataset
    1 : 0,  # Buildings   -> Buildings
    2 : 0,  # Fences      -> Buildings
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
    17 : 1, # GuardRail   -> Misc
    19 : 1, # Static      -> Misc
    20 : 1, # Dynamic     -> Misc
    21 : 6, # Water       -> Water
    22 : 4  # Terrain     -> Vegetation
}
__carla_to_diminished_ids = {
    # 0, 4, 5, 12, 13 are assumed to be absent in the dataset
    1 : 2,  # Buildings   -> Non-drivable
    2 : 2,  # Fences      -> Non-drivable
    3 : 2,  # Other       -> Non-drivable
    6 : 1,  # RoadLine    -> Drivable
    7 : 1,  # Road        -> Drivable
    8 : 2,  # SideWalk    -> Non-drivable
    9 : 2,  # Vegetation  -> Non-drivable
    10 : 0, # Vehicles    -> Vehicles
    11 : 2, # Walls       -> Non-drivable
    12 : 2, # TrafficSign -> Non-drivable
    14 : 2, # Ground      -> Non-drivable
    15 : 2, # Bridge      -> Non-drivable
    16 : 2, # RailTrack   -> Non-drivable
    17 : 2, # GuardRail   -> Non-drivable
    19 : 2, # Static      -> Non-drivable
    20 : 2, # Dynamic     -> Non-drivable
    21 : 2, # Water       -> Non-drivable
    22 : 2  # Terrain     -> Non-drivable
}

def convert_semantic_classes(semantic_ids : np.ndarray, target_classes: str, mask: np.ndarray = None) -> np.ndarray:
    """
    converts a HxW array of carla semantic IDs to a predefined subset ('ours' or 'diminished').
    if 'carla' or an unknown string is passed as target, the function retuns the exact input
    """
    assert len(semantic_ids.shape) == 2, f'expected HxW, got {semantic_ids.shape}'
    # if target_classes is unknown (or 'carla') return identity
    target_semantics = semantic_ids
    if target_classes == 'ours':
        target_semantics = np.zeros_like(semantic_ids)
        for carla_id, our_id in __carla_to_our_ids.items():
            target_semantics[semantic_ids == carla_id] = our_id
    elif target_classes == 'ours+mask':
        target_semantics = np.zeros_like(semantic_ids)
        for carla_id, our_id in __carla_to_our_ids.items():
            target_semantics[semantic_ids == carla_id] = our_id
        # add mask as label
        target_semantics[mask == 1] = 7
    elif target_classes == 'diminished':
        target_semantics = np.zeros_like(semantic_ids)
        for carla_id, our_id in __carla_to_diminished_ids.items():
            target_semantics[semantic_ids == carla_id] = our_id
    return target_semantics

def convert_semantics_to_rgb(semantic_ids : torch.Tensor, semantic_classes: str) -> np.ndarray:
    """
    converts a tensor of semantic IDs to an RGB image of shape HxWx3
    """
    assert len(semantic_ids.shape) == 2, f'expected HxW, got {semantic_ids.shape}'
    if semantic_classes == 'carla':
        semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                         semantic_ids.shape[1], 3),
                                  dtype=np.uint8)
        for sid, cityscapes_rgb in __carla_to_rgb_palette.items():
            semantic_rgb[semantic_ids == sid] = cityscapes_rgb
    elif semantic_classes == 'ours':
        semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                         semantic_ids.shape[1], 3),
                                  dtype=np.uint8)
        for sid, cityscapes_rgb in __ours_to_rgb_palette.items():
            semantic_rgb[semantic_ids == sid] = cityscapes_rgb
    elif semantic_classes == 'ours+mask':
        semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                         semantic_ids.shape[1], 3),
                                  dtype=np.uint8)
        for sid, cityscapes_rgb in __oursplusmask_to_rgb_palette.items():
            semantic_rgb[semantic_ids == sid] = cityscapes_rgb
    elif semantic_classes == 'diminished':
        semantic_rgb = np.ndarray(shape=(semantic_ids.shape[0],
                                         semantic_ids.shape[1], 3),
                                  dtype=np.uint8)
        for sid, cityscapes_rgb in __diminished_to_rgb_palette.items():
            semantic_rgb[semantic_ids == sid] = cityscapes_rgb
    else:
        return None
    return semantic_rgb