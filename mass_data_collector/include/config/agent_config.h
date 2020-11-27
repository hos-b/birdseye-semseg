#ifndef __MASS_AGENT_CONFIG__
#define __MASS_AGENT_CONFIG__
#include <cmath>
#include <opencv2/core/types.hpp>
#include <unordered_map>

namespace config
{
	// stuff
	extern const unsigned int kPollInterval; // miliseconds

	// CARLA's semantic IDs
	extern const unsigned int kCARLAUnknownSemanticID;
	extern const unsigned int kCARLABuildingSemanticID;
	extern const unsigned int kCARLAFenceSemanticID;
	extern const unsigned int kCARLAOtherSemanticID;
	extern const unsigned int kCARLAPedestrianSemanticID;
	extern const unsigned int kCARLAPoleSemanticID;
	extern const unsigned int kCARLARoadLineSemanticID;
	extern const unsigned int kCARLARoadSemanticID;
	extern const unsigned int kCARLASideWalkSemanticID;
	extern const unsigned int kCARLAVegetationSemanticID;
	extern const unsigned int kCARLAVehiclesSemanticID;
	extern const unsigned int kCARLAWallSemanticID;
	extern const unsigned int kCARLATrafficSignSemanticID;
	extern const unsigned int kCARLASkySemanticID;
	extern const unsigned int kCARLAGroundSemanticID;
	extern const unsigned int kCARLABridgeSemanticID;
	extern const unsigned int kCARLARailTrackSemanticID;
	extern const unsigned int kCARLAGuardRailSemanticID;
	extern const unsigned int kCARLATrafficLightSemanticID;
	extern const unsigned int kCARLAStaticSemanticID;
	extern const unsigned int kCARLADynamicSemanticID;
	extern const unsigned int kCARLAWaterSemanticID;
	extern const unsigned int kCARLATerrainSemanticID;

	// our semantic IDs
	extern const unsigned int kMassUnknownSemanticID;
	extern const unsigned int kMassBuildingSemanticID;
	extern const unsigned int kMassStaticSemanticID;
	extern const unsigned int kMassDynamicSemanticID;
	extern const unsigned int kMassRoadLineSemanticID;
	extern const unsigned int kMassRoadSemanticID;
	extern const unsigned int kMassSideWalkSemanticID;
	extern const unsigned int kMassVehiclesSemanticID;
	extern const unsigned int kMassOtherSemanticID;
	extern const unsigned int kMassSkySemanticID;
	extern const unsigned int kMassTerrainSemanticID;

	// hash maps
	extern const std::unordered_map<unsigned int, unsigned int> semantic_conversion_map;
	extern const std::unordered_map<unsigned int, cv::Vec3b> semantic_palette_map;
	extern const std::unordered_map<unsigned int, bool> fileterd_semantics;
} // namespace config

#endif