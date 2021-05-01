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
        /* kCARLAFenceSemanticID : */           true,  // <-- who needs fences TODO: false? 
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
        /* kCARLARailTrackSemanticID : */       true,  // TODO: true ?
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
    // perfect for town_0
    const std::unordered_map<int, bool> town0_restricted_roads ({
        {499, true}, // junction lane leading to tunnel
        {551, true}, // junction lane leading to tunnel
        {510, true}, // junction lane leading to tunnel
        {519, true}, // junction lane leading to tunnel
        {1166, true},// junction lane leading to tunnel
        {526, true}, // junction after the tunnel
        {1165, true},// junction after the tunnel
        {1158, true},// junction after the tunnel
        {1156, true},// junction after the tunnel
        {65, true},  // tunnel
        {58, true},  // gas station
    });

} // namespace config