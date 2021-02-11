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
	extern const unsigned int kCARLAUnlabeledSemanticID; // Elements that have not been categorized are considered Unlabeled. This category is meant to be empty or at least contain elements with no collisions.
	extern const unsigned int kCARLABuildingSemanticID; // Buildings like houses, skyscrapers,... and the elements attached to them. E.g. air conditioners, scaffolding, awning or ladders and much more.
	extern const unsigned int kCARLAFenceSemanticID; // Barriers, railing, or other upright structures. Basically wood or wire assemblies that enclose an area of ground.
	extern const unsigned int kCARLAOtherSemanticID; // Everything that does not belong to any other category.
	extern const unsigned int kCARLAPedestrianSemanticID; // Humans that walk or ride/drive any kind of vehicle or mobility system. E.g. bicycles or scooters, skateboards, horses, roller-blades, wheel-chairs, etc.
	extern const unsigned int kCARLAPoleSemanticID; // Small mainly vertically oriented pole. If the pole has a horizontal part (often for traffic light poles) this is also considered pole. E.g. sign pole, traffic light poles.
	extern const unsigned int kCARLARoadLineSemanticID; // The markings on the road.
	extern const unsigned int kCARLARoadSemanticID; // Part of ground on which cars usually drive. E.g. lanes in any directions, and streets.
	extern const unsigned int kCARLASideWalkSemanticID; // Part of ground designated for pedestrians or cyclists. Delimited from the road by some obstacle (such as curbs or poles), not only by markings. This label includes a possibly delimiting curb, traffic islands (the walkable part), and pedestrian zones.
	extern const unsigned int kCARLAVegetationSemanticID; // Trees, hedges, all kinds of vertical vegetation. Ground-level vegetation is considered Terrain.
	extern const unsigned int kCARLAVehiclesSemanticID; // Cars, vans, trucks, motorcycles, bikes, buses, trains.
	extern const unsigned int kCARLAWallSemanticID; // Individual standing walls. Not part of a building.
	extern const unsigned int kCARLATrafficSignSemanticID; // Signs installed by the state/city authority, usually for traffic regulation. This category does not include the poles where signs are attached to. E.g. traffic- signs, parking signs, direction signs...
	extern const unsigned int kCARLASkySemanticID; // Open sky. Includes clouds and the sun.
	extern const unsigned int kCARLAGroundSemanticID; // Any horizontal ground-level structures that does not match any other category. For example areas shared by vehicles and pedestrians, or flat roundabouts delimited from the road by a curb.
	extern const unsigned int kCARLABridgeSemanticID; // Only the structure of the bridge. Fences, people, vehicles, an other elements on top of it are labeled separately.
	extern const unsigned int kCARLARailTrackSemanticID; // All kind of rail tracks that are non-drivable by cars. E.g. subway and train rail tracks.
	extern const unsigned int kCARLAGuardRailSemanticID; // All types of guard rails/crash barriers.
	extern const unsigned int kCARLATrafficLightSemanticID; // Traffic light boxes without their poles.
	extern const unsigned int kCARLAStaticSemanticID; // Elements in the scene and props that are immovable. E.g. fire hydrants, fixed benches, fountains, bus stops, etc.
	extern const unsigned int kCARLADynamicSemanticID; // Elements whose position is susceptible to change over time. E.g. Movable trash bins, buggies, bags, wheelchairs, animals, etc.
	extern const unsigned int kCARLAWaterSemanticID; // Horizontal water surfaces. E.g. Lakes, sea, rivers.
	extern const unsigned int kCARLATerrainSemanticID; // Grass, ground-level vegetation, soil or sand. These areas are not meant to be driven on. This label includes a possibly delimiting curb.

	// our semantic IDs
	extern const unsigned int kMassUnlabeledSemanticID;
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

	// hash maps for semantic ids
	extern const std::unordered_map<unsigned int, unsigned int> semantic_conversion_map;
	extern const std::unordered_map<unsigned int, cv::Vec3b> carla_to_cityscapes_palette_map;
	extern const std::unordered_map<unsigned int, bool> fileterd_semantics;
	extern const std::unordered_map<unsigned int, double> semantic_weight;

	// hash maps for restricted areas on map
	extern const std::unordered_map<int, bool> town0_restricted_roads;
} // namespace config

#endif