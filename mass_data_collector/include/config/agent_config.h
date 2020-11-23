#ifndef __MASS_AGENT_CONFIG__
#define __MASS_AGENT_CONFIG__
#include <cmath>
#include <unordered_map>

namespace config
{
	static constexpr float kMinimumAgentDistance = 0.7f; // meters
	// additional BEV camera formation
	static constexpr float kCamLRHover = 2.0f; // meters
	static constexpr float kCamFBHover = 2.0f; // meters

	// boundaries, sizes
	static constexpr double kPointCloudMaxLocalX = 10.0; // meters
	static constexpr double kPointCloudMaxLocalY = 10.0; // meters
	static constexpr unsigned int kSemanticBEVRows = 1000; // pixels
	static constexpr unsigned int kSemanticBEVCols = 1000; // pixels

	// stuff
	static constexpr unsigned int kPollInterval = 10; // milisecond

	// semantic color of the agent
	static constexpr unsigned int kVehicleSemanticR = 0;
	static constexpr unsigned int kVehicleSemanticG = 0;
	static constexpr unsigned int kVehicleSemanticB = 142;

	// CARLA's semantic IDs
	static constexpr unsigned int kCARLAUnknownSemanticID = 0;
	static constexpr unsigned int kCARLABuildingSemanticID = 1;
	static constexpr unsigned int kCARLAFenceSemanticID = 2;
	static constexpr unsigned int kCARLAOtherSemanticID = 3;
	static constexpr unsigned int kCARLAPedestrianSemanticID = 4;
	static constexpr unsigned int kCARLAPoleSemanticID = 5;
	static constexpr unsigned int kCARLARoadLineSemanticID = 6;
	static constexpr unsigned int kCARLARoadSemanticID = 7;
	static constexpr unsigned int kCARLASideWalkSemanticID = 8;
	static constexpr unsigned int kCARLAVegetationSemanticID = 9;
	static constexpr unsigned int kCARLAVehiclesSemanticID = 10;
	static constexpr unsigned int kCARLAWallSemanticID = 11;
	static constexpr unsigned int kCARLATrafficSignSemanticID = 12;
	static constexpr unsigned int kCARLASkySemanticID = 13;
	static constexpr unsigned int kCARLAGroundSemanticID = 14;
	static constexpr unsigned int kCARLABridgeSemanticID = 15;
	static constexpr unsigned int kCARLARailTrackSemanticID = 16;
	static constexpr unsigned int kCARLAGuardRailSemanticID = 17;
	static constexpr unsigned int kCARLATrafficLightSemanticID = 18;
	static constexpr unsigned int kCARLAStaticSemanticID = 19;
	static constexpr unsigned int kCARLADynamicSemanticID = 20;
	static constexpr unsigned int kCARLAWaterSemanticID = 21;
	static constexpr unsigned int kCARLATerrainSemanticID = 22;

	// our semantic IDs
	static constexpr unsigned int kMassUnknownSemanticID = 0;
	static constexpr unsigned int kMassBuildingSemanticID = 1;
	static constexpr unsigned int kMassStaticSemanticID = 2;
	static constexpr unsigned int kMassDynamicSemanticID = 3;
	static constexpr unsigned int kMassRoadLineSemanticID = 4;
	static constexpr unsigned int kMassRoadSemanticID = 5;
	static constexpr unsigned int kMassSideWalkSemanticID = 6;
	static constexpr unsigned int kMassVehiclesSemanticID = 7;
	static constexpr unsigned int kMassOtherSemanticID = 8;
	static constexpr unsigned int kMassSkySemanticID = 9;
	static constexpr unsigned int kMassTerrainSemanticID = 10;

	static const std::unordered_map<unsigned int, unsigned int> semantic_conversion_map({  // NOLINT
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

} // namespace config

#endif