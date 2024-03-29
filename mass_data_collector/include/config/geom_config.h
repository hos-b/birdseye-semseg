#ifndef __MASS_GEOM_CONFIG__
#define __MASS_GEOM_CONFIG__

namespace config
{
	const extern float kToRadians;
	const extern float kToDegrees;
	const extern double kPixelDistanceThreshold;

	// additional BEV camera formation
	const extern float kCamLRHover; // meters
	const extern float kCamFBHover; // meters

	// boundaries, sizes
	const extern unsigned int kMaxAgentDistance; // centimeters

	// map kd-tree
	const extern float kWaypointGenerationDistance;
	const extern unsigned int kWaypointKDTreeBins;

	// agent positioning
	const extern float kMinDistCoeff;

} // namespace config

#endif