#ifndef __CARLA_TRANSFORM_CONVERSIONS_H__
#define __CARLA_TRANSFORM_CONVERSIONS_H__

#include <carla/geom/Transform.h>

carla::geom::Transform ApplyTransform(carla::geom::Transform tf,
                                      float delta_x, float delta_y,
                                      float delta_pitch, float delta_yaw);

#endif