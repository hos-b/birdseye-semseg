#ifndef __MASS_AGENT_CONFIG__
#define __MASS_AGENT_CONFIG__
#include <cmath>

namespace config
{
  static constexpr float kToRadians  = static_cast<float>(M_PI / 180.0f);
  static constexpr float kToDegrees  = static_cast<float>(180.0f / M_PI);
  static constexpr float kMinimumAgentDistance = 0.7; // meters
  static constexpr float kDepthMax = 256;
  static constexpr float kDepthRange = 1000;
} // namespace config

#endif