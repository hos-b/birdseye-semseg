#include "config/agent_config.h"
#include <opencv2/core/matx.hpp>
#include <vector>

/* turned all into extern to speed up compilation */

namespace config
{
    constexpr unsigned int kPollInterval = 10; // miliseconds

    // carla semantic IDs
    constexpr unsigned int kCARLAUnknownSemanticID = 0;
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

    // our filtered IDs
    constexpr unsigned int kMassUnknownSemanticID = 0;
    constexpr unsigned int kMassBuildingSemanticID = 1;
    constexpr unsigned int kMassStaticSemanticID = 2;
    constexpr unsigned int kMassDynamicSemanticID = 3;
    constexpr unsigned int kMassRoadLineSemanticID = 4;
    constexpr unsigned int kMassRoadSemanticID = 5;
    constexpr unsigned int kMassSideWalkSemanticID = 6;
    constexpr unsigned int kMassVehiclesSemanticID = 7;
    constexpr unsigned int kMassOtherSemanticID = 8;
    constexpr unsigned int kMassSkySemanticID = 9;
    constexpr unsigned int kMassTerrainSemanticID = 10;

    // converting from CARLA to ours
    const std::unordered_map<unsigned int, unsigned int> semantic_conversion_map({ // NOLINT
		{kCARLAUnknownSemanticID, kMassUnknownSemanticID},
		{kCARLABuildingSemanticID, kMassBuildingSemanticID},
		{kCARLAFenceSemanticID, kMassBuildingSemanticID},
		{kCARLAOtherSemanticID, kMassOtherSemanticID},
		{kCARLAPedestrianSemanticID, kMassDynamicSemanticID},
		{kCARLAPoleSemanticID, kMassStaticSemanticID},
		{kCARLARoadLineSemanticID, kMassRoadLineSemanticID},
		{kCARLARoadSemanticID, kMassRoadSemanticID},
		{kCARLASideWalkSemanticID, kMassSideWalkSemanticID},
		{kCARLAVegetationSemanticID, kMassOtherSemanticID},
		{kCARLAVehiclesSemanticID, kMassVehiclesSemanticID},
		{kCARLAWallSemanticID, kMassStaticSemanticID},
		{kCARLATrafficSignSemanticID, kMassStaticSemanticID},
		{kCARLASkySemanticID, kMassSkySemanticID},
		{kCARLAGroundSemanticID, kMassRoadSemanticID},
		{kCARLABridgeSemanticID, kMassBuildingSemanticID},
		{kCARLARailTrackSemanticID, kMassStaticSemanticID},
		{kCARLAGuardRailSemanticID, kMassStaticSemanticID},
		{kCARLATrafficLightSemanticID, kMassStaticSemanticID},
		{kCARLAStaticSemanticID, kMassStaticSemanticID},
		{kCARLADynamicSemanticID, kMassDynamicSemanticID},
		{kCARLAWaterSemanticID, kMassTerrainSemanticID},
		{kCARLATerrainSemanticID, kMassTerrainSemanticID}
    });
    // CARLA's semantic id -> BGR (Cityescapes?)
    const std::unordered_map<unsigned int, cv::Vec3b> semantic_palette_map({ // NOLINT
        {kCARLAUnknownSemanticID, cv::Vec3b(0, 0, 0)},
        {kCARLABuildingSemanticID, cv::Vec3b(70, 70, 70)},
        {kCARLAFenceSemanticID, cv::Vec3b(40, 40, 100)},
        {kCARLAOtherSemanticID, cv::Vec3b(80, 90, 55)},
        {kCARLAPedestrianSemanticID, cv::Vec3b(60,  20,  220)},
        {kCARLAPoleSemanticID, cv::Vec3b(153, 153, 153)},
        {kCARLARoadLineSemanticID, cv::Vec3b(50, 234, 157)},
        {kCARLARoadSemanticID, cv::Vec3b(128, 64, 128)},
        {kCARLASideWalkSemanticID, cv::Vec3b(232, 35, 244)},
        {kCARLAVegetationSemanticID, cv::Vec3b(35, 142, 107)},
        {kCARLAVehiclesSemanticID, cv::Vec3b(142, 0, 0)},
        {kCARLAWallSemanticID, cv::Vec3b(156, 102, 102)},
        {kCARLATrafficSignSemanticID, cv::Vec3b(0, 220, 220)},
        {kCARLASkySemanticID, cv::Vec3b(180, 130, 70)},
        {kCARLAGroundSemanticID, cv::Vec3b(81, 0, 81)},
        {kCARLABridgeSemanticID, cv::Vec3b(100, 100, 150)},
        {kCARLARailTrackSemanticID, cv::Vec3b(140, 150, 230)},
        {kCARLAGuardRailSemanticID, cv::Vec3b(180, 165, 180)},
        {kCARLATrafficLightSemanticID, cv::Vec3b(30, 170, 250)},
        {kCARLAStaticSemanticID, cv::Vec3b(160, 190, 110)},
        {kCARLADynamicSemanticID, cv::Vec3b(50, 120, 170)},
        {kCARLAWaterSemanticID, cv::Vec3b(150, 60, 45)},
        {kCARLATerrainSemanticID, cv::Vec3b(100, 170, 145)}
    });
    const std::unordered_map<unsigned int, bool> fileterd_semantics({ // NOLINT
        {kCARLAUnknownSemanticID, false},
        {kCARLABuildingSemanticID, false},
        {kCARLAFenceSemanticID, false},
        {kCARLAOtherSemanticID, false},
        {kCARLAPedestrianSemanticID, false},
        {kCARLAPoleSemanticID, true},
        {kCARLARoadLineSemanticID, false},
        {kCARLARoadSemanticID, false},
        {kCARLASideWalkSemanticID, false},
        {kCARLAVegetationSemanticID, false},
        {kCARLAVehiclesSemanticID, false},
        {kCARLAWallSemanticID, false},
        {kCARLATrafficSignSemanticID, false},
        {kCARLASkySemanticID, true},
        {kCARLAGroundSemanticID, false},
        {kCARLABridgeSemanticID, false},
        {kCARLARailTrackSemanticID, false},
        {kCARLAGuardRailSemanticID, false},
        {kCARLATrafficLightSemanticID, true},
        {kCARLAStaticSemanticID, false},
        {kCARLADynamicSemanticID, false},
        {kCARLAWaterSemanticID, false},
        {kCARLATerrainSemanticID, false}
    });
} // namespace config