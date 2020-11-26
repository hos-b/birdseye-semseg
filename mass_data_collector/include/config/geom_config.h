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
	const extern double kPointCloudMaxLocalX; // meters
	const extern double kPointCloudMaxLocalY; // meters
	const extern unsigned int kSemanticBEVRows; // pixels
	const extern unsigned int kSemanticBEVCols; // pixels
	const extern unsigned int kMaxAgentDistance; // centimeters
	const extern float kMinimumAgentDistance; // meters


} // namespace config

#endif