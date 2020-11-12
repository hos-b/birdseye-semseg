#ifndef __MASS_GEOM_CONFIG__
#define __MASS_GEOM_CONFIG__
#include <cmath>

namespace config
{
	static constexpr float kToRadians  = static_cast<float>(M_PI / 180.0f);
	static constexpr float kToDegrees  = static_cast<float>(180.0f / M_PI);
	static constexpr double kPixelDistanceThreshold  = 1.5;
} // namespace config

#endif