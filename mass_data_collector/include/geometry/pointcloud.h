#ifndef __MASS_PCLOUD_H__
#define __MASS_PCLOUD_H__

#include "Eigen/Dense"
#include <variant>
#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>

namespace geom 
{

class SemanticPoint3D
{
	enum cs : unsigned int {
		GLOBAL_COORDINATES      = 0b000001,
		CAMERA_COORDINATES      = 0b100000,
		// if on camera coordinates, these help choose the agent
		CLEAR_AGENT_ID          = 0b100000,
		SELECT_AGENT_ID         = 0b011111
	};
public:
	[[nodiscard]] const Eigen::Vector3f& position() const;
private:
	Eigen::Vector3f pos_;
	// either semantic id or cityscapes rgb
	std::variant<unsigned char, Eigen::Vector3f> data_;
};

class SemanticCloud
{
public:
	SemanticCloud();
private:
	open3d::geometry::PointCloud cloud_;
};

} // namespace geom
#endif