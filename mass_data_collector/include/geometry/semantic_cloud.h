#ifndef __MASS_PCLOUD_H__
#define __MASS_PCLOUD_H__

#include <functional>
#include <memory>
#include <variant>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>
#include <nanoflann.hpp>

#include "geometry/camera_geomtry.h"

namespace geom 
{

class SemanticCloud
{
public:
	SemanticCloud(double max_x, double max_y, size_t img_rows, size_t img_cols);
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
								cv::Mat semantic,
								cv::Mat depth);
	pcl::PointCloud<pcl::PointXYZRGB> GetMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry);
	std::pair <cv::Mat, cv::Mat> GetSemanticBEV(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
						   double pixel_limit, double mask_dist_threshold);
	void SaveTargetCloud(const std::string& path);
	void FilterCloud();

	// kd-tree stuff
	inline size_t kdtree_get_point_count() const { return target_cloud_.points.size(); } // NOLINT
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const { // NOLINT
		if (dim == 0) {
			return target_cloud_.points[idx].x; // NOLINT
		} else {								// NOLINT
			return target_cloud_.points[idx].y; // NOLINT
		}
	}
	template<class BBox>
	bool kdtree_get_bbox(BBox& /* bb */) const { return false; }

private:
	nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, SemanticCloud>,
										SemanticCloud, 2> *kd_tree_;
	pcl::PointCloud<pcl::PointXYZRGB> target_cloud_;
	double point_max_y_;
	double point_max_x_;
	size_t image_cols_;
	size_t image_rows_;
	std::function<size_t(const std::pair<double, double>&)> xy_hash_;
	std::function<size_t(const Eigen::Vector3i&)> semantic_color_hash_;
};

} // namespace geom
#endif