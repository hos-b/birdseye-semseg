#include "config/agent_config.h"
#include <opencv2/core/matx.hpp>
#include <vector>

/* turned all into extern to speed up compilation */

namespace config
{
    constexpr unsigned int kPollInterval = 10; // miliseconds

    // carla semantic IDs
    constexpr unsigned int kCARLAUnlabeledSemanticID = 0;
    constexpr unsigned int kCARLABuildingSemanticID = 1;
    constexpr unsigned int kCARLAFenceSemanticID = 2;
    constexpr unsigned int kCARLAOtherSemanticID = 3;
    constexpr unsigned int kCARLAPedestrianSemanticID = 4;
    constexpr unsigned int kCARLAPoleSemanticID = 5;
    constexpr unsigned int kCARLARoadLineSemanticID = 6;
    constexpr unsigned int kCARLARoadSemanticID = 7;
    constexpr unsigned int kCARLASideWalkSemanticID = 8;
    constexpr unsigned int kCARLAVegetationSemanticID = 9;
    constexpr unsigned int kCARLAVehiclesSemanticID = 10;
    constexpr unsigned int kCARLAWallSemanticID = 11;
    constexpr unsigned int kCARLATrafficSignSemanticID = 12;
    constexpr unsigned int kCARLASkySemanticID = 13;
    constexpr unsigned int kCARLAGroundSemanticID = 14;
    constexpr unsigned int kCARLABridgeSemanticID = 15;
    constexpr unsigned int kCARLARailTrackSemanticID = 16;
    constexpr unsigned int kCARLAGuardRailSemanticID = 17;
    constexpr unsigned int kCARLATrafficLightSemanticID = 18;
    constexpr unsigned int kCARLAStaticSemanticID = 19;
    constexpr unsigned int kCARLADynamicSemanticID = 20;
    constexpr unsigned int kCARLAWaterSemanticID = 21;
    constexpr unsigned int kCARLATerrainSemanticID = 22;
    constexpr unsigned int kCARLASemanticClassCount = 23;

    // CARLA's semantic id -> Cityescapes BGR
    const cv::Vec3b carla_to_cityscapes_palette[] {
        cv::Vec3b(0, 0, 0),
        cv::Vec3b(70, 70, 70),
        cv::Vec3b(40, 40, 100),
        cv::Vec3b(80, 90, 55),
        cv::Vec3b(60,  20,  220),
        cv::Vec3b(153, 153, 153),
        cv::Vec3b(50, 234, 157),
        cv::Vec3b(128, 64, 128),
        cv::Vec3b(232, 35, 244),
        cv::Vec3b(35, 142, 107),
        cv::Vec3b(142, 0, 0),
        cv::Vec3b(156, 102, 102),
        cv::Vec3b(0, 220, 220),
        cv::Vec3b(180, 130, 70),
        cv::Vec3b(81, 0, 81),
        cv::Vec3b(100, 100, 150),
        cv::Vec3b(140, 150, 230),
        cv::Vec3b(180, 165, 180),
        cv::Vec3b(30, 170, 250),
        cv::Vec3b(160, 190, 110),
        cv::Vec3b(50, 120, 170),
        cv::Vec3b(150, 60, 45),
        cv::Vec3b(100, 170, 145),
    };
    constexpr bool filtered_semantics[] {
        /* kCARLAUnlabeledSemanticID : */       true,  // <-- hopefully empty
        /* kCARLABuildingSemanticID : */        false,
        /* kCARLAFenceSemanticID : */           true,  // <-- who needs fences? probably very thin 
        /* kCARLAOtherSemanticID : */           false,
        /* kCARLAPedestrianSemanticID : */      false,
        /* kCARLAPoleSemanticID : */            true,  // <-- over-hanging structure + kinda irrelevant
        /* kCARLARoadLineSemanticID : */        false,
        /* kCARLARoadSemanticID : */            false,
        /* kCARLASideWalkSemanticID : */        false,
        /* kCARLAVegetationSemanticID : */      false,
        /* kCARLAVehiclesSemanticID : */        false,
        /* kCARLAWallSemanticID : */            false,
        /* kCARLATrafficSignSemanticID : */     true,  // <-- not important for top-down
        /* kCARLASkySemanticID : */             true,  // <-- won't happen anyway but meh
        /* kCARLAGroundSemanticID : */          false,
        /* kCARLABridgeSemanticID : */          false,
        /* kCARLARailTrackSemanticID : */       true,  // <-- helps data collection in town3
        /* kCARLAGuardRailSemanticID : */       true,  // <-- not important for top-down
        /* kCARLATrafficLightSemanticID : */    true,  // <-- over-hanging structure
        /* kCARLAStaticSemanticID : */          false,
        /* kCARLADynamicSemanticID : */         false,
        /* kCARLAWaterSemanticID : */           false,
        /* kCARLATerrainSemanticID : */         false
    };
    const double semantic_weight[23] {
        static_cast<double>(filtered_semantics[kCARLAUnlabeledSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLABuildingSemanticID]) * 2.0,
        static_cast<double>(filtered_semantics[kCARLAFenceSemanticID]) * 0.0, // TODO: allow ?
        static_cast<double>(filtered_semantics[kCARLAOtherSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLAPedestrianSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLAPoleSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLARoadLineSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLARoadSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLASideWalkSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLAVegetationSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLAVehiclesSemanticID]) * 6.0,
        static_cast<double>(filtered_semantics[kCARLAWallSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLATrafficSignSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLASkySemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLAGroundSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLABridgeSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLARailTrackSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLAGuardRailSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLATrafficLightSemanticID]) * 0.0,
        static_cast<double>(filtered_semantics[kCARLAStaticSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLADynamicSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLAWaterSemanticID]) * 1.0,
        static_cast<double>(filtered_semantics[kCARLATerrainSemanticID]) * 1.0
    };
    // town1 roads to avoid
    const std::unordered_map<int, bool> town1_restricted_roads ({
    });
    // town2 roads to avoid
    const std::unordered_map<int, bool> town2_restricted_roads ({
    });
    // town3 roads to avoid
    const std::unordered_map<int, bool> town3_restricted_roads ({
        {65, true},  // tunnel
        {526, true}, // junction after the tunnel
        {510, true}, // junction lane leading to tunnel
        {499, true}, // junction lane leading to tunnel
        {551, true}, // junction lane leading to tunnel
        {533, true}, // junction lane leading to tunnel
        {519, true}, // junction lane leading to tunnel
        {1166, true},// junction lane leading to tunnel
        {1165, true},// junction after the tunnel
        {1158, true},// junction after the tunnel
        {1156, true},// junction after the tunnel

        {58, true},  // gas station
        {1020, true},  // before gas station
        {1933, true},  // after gas station
    });
    // town4 roads to avoid
    const std::unordered_map<int, bool> town4_restricted_roads ({
        {47, true}, // overpass

        {880, true}, // gas station
        {886, true}, // gas station
        {477, true}, // gas station
        {467, true}, // gas station
        {468, true}, // gas station

    });
    // town5 roads to avoid
    const std::unordered_map<int, bool> town5_restricted_roads ({
        {39, true}, // overhanging building

        {3, true}, // leading to underpass
        {9, true}, // leading to underpass
        {10, true}, // underpass
        {11, true}, // underpass
        {24, true}, // underpass
        {1931, true}, // underpass junction
        {1940, true}, // underpass junction
        {1946, true}, // underpass junction
        {1957, true}, // underpass junction
        {1985, true}, // underpass junction
        {1986, true}  // underpass junction
    });
    // town6 roads to avoid
    const std::unordered_map<int, bool> town6_restricted_roads ({

    });
    // town7 roads to avoid
    const std::unordered_map<int, bool> town7_restricted_roads ({

    });
    // town10 roads to avoid
    const std::unordered_map<int, bool> town10_restricted_roads ({
        {8, true}, // under tram stop
    });
    const std::unordered_map<int, bool>* restricted_roads[] = {
        nullptr,
        &town1_restricted_roads,
        &town2_restricted_roads,
        &town3_restricted_roads,
        &town4_restricted_roads,
        &town5_restricted_roads,
        &town6_restricted_roads,
        &town7_restricted_roads,
        nullptr,
        nullptr,
        &town10_restricted_roads
    };
    const std::unordered_map<int, std::string> town_map_strings ({
        {1, "/Game/Carla/Maps/Town01"},
        {2, "/Game/Carla/Maps/Town02"},
        {3, "/Game/Carla/Maps/Town03"},
        {4, "/Game/Carla/Maps/Town04"},
        {5, "/Game/Carla/Maps/Town05"},
        {6, "/Game/Carla/Maps/Town06"},
        {7, "/Game/Carla/Maps/Town07"},
        {8, "does not exist"},
        {9, "does not exist"},
        {10, "/Game/Carla/Maps/Town10HD"}
    });
} // namespace config