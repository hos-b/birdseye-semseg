#ifndef __MASS_AGENT_CONFIG__
#define __MASS_AGENT_CONFIG__
#include <cmath>

namespace config
{
	static constexpr float kMinimumAgentDistance = 0.7f; // meters
	// additional BEV camera formation
	static constexpr float kCamLRHover = 2.0f; // meters
	static constexpr float kCamFBHover = 2.0f; // meters
	static constexpr unsigned int kFrameVizInterval = 5;
} // namespace config

#endif