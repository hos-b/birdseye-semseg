#ifndef __AIS_AGENT_UTILS__
#define __AIS_AGENT_UTILS__

namespace freicar
{
namespace config
{
// spectator camera smoothing factor
static const float kSpectatorCamSmoothFactor = 0.85f;

namespace {
  inline void ConfigPlaceHolder() {
    (void)config::kSpectatorCamSmoothFactor;
  }
} // namespace

} // namespace config
} // namespace aiscar

#endif