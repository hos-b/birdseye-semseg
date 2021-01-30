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
	constexpr unsigned int kMaxAgentDistance = 400; // centimeters


} // namespace config
