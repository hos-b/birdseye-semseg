#include "config/geom_config.h"

# define M_PI		3.14159265358979323846	/* pi */

namespace config
{
	constexpr float kToRadians = static_cast<float>(M_PI / 180.0f);
	constexpr float kToDegrees = static_cast<float>(180.0f / M_PI);
	constexpr double kPixelDistanceThreshold  = 1.5;

	// additional BEV camera formation
	constexpr float kCamLRHover = 5.0f; // meters
	constexpr float kCamFBHover = 5.0f; // meters

	// boundaries, sizes
	constexpr double kPointCloudMaxLocalX = 10.0; // meters
	constexpr double kPointCloudMaxLocalY = 10.0; // meters
	constexpr unsigned int kSemanticBEVRows = 1000; // pixels
	constexpr unsigned int kSemanticBEVCols = 1000; // pixels
	constexpr unsigned int kMaxAgentDistance = 400; // centimeters


} // namespace config
