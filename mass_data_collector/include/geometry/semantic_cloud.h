#ifndef __MASS_PCLOUD_H__
#define __MASS_PCLOUD_H__

#include <memory>

#include <variant>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#ifndef __clang_analyzer__
#include <pcl/point_types.h>  // NOLINT
#endif
#include <opencv2/opencv.hpp>

#include "geometry/camera_geomtry.h"

namespace geom 
{

class SemanticPoint3D
{
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
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   const Eigen::Matrix4d& car_transform,
							   cv::Mat semantic,
							   cv::Mat depth);
	void MaskOutlierPoints(std::shared_ptr<geom::CameraGeometry> geometry,
						   const Eigen::Matrix4d& car_transform);
	void SaveCloud(const std::string& path);
	[[nodiscard]] cv::Mat GetBEV() const;

private:
	pcl::PointCloud<pcl::PointXYZRGB> cloud_;
};

} // namespace geom
#endif