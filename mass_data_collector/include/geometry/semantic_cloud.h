#ifndef __MASS_PCLOUD_H__
#define __MASS_PCLOUD_H__

#include <functional>
#include <memory>
#include <variant>

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "geometry/camera_geomtry.h"

namespace geom 
{

class SemanticCloud // NOLINT
{
using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, SemanticCloud>,
													 SemanticCloud, 2>;

public:
	SemanticCloud(double max_x, double max_y, size_t img_rows, size_t img_cols);
	~SemanticCloud();
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
								cv::Mat semantic,
								cv::Mat depth);
	std::pair <cv::Mat, cv::Mat> GetSemanticBEV(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
								double pixel_limit, double vehicle_width, double vehicle_length);
	cv::Mat GetBEVMask();
	void ProcessCloud();
	static cv::Mat ConvertToCityScapesPallete(cv::Mat semantic_ids);

	// debug functions
	void PrintBoundaries();
	void SaveCloud(const std::string& path);
	std::tuple<double, double, double, double> GetVehicleBoundary();
	void SaveMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
						 const std::string& path, double pixel_limit);

	// mandatory kd-tree stuff
	[[nodiscard]] inline size_t kdtree_get_point_count() const { return target_cloud_.points.size(); }
	[[nodiscard]] inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
		if (dim == 0) {
			return target_cloud_.points[idx].x;
		}
		return target_cloud_.points[idx].y;
	}
	template<class BBox>
	bool kdtree_get_bbox(BBox& /* bb */) const { return false; }
private:
	size_t GetMajorityVote(const std::vector<size_t>& knn_result);
	std::pair<std::vector<size_t>, std::vector<double>> FindClosestPoints(double knn_x, double knn_y, size_t num_results);

	// members
	std::unique_ptr<KDTree2D> kd_tree_;
	pcl::PointCloud<pcl::PointXYZL> target_cloud_;
	double point_max_y_;
	double point_max_x_;
	size_t image_cols_;
	size_t image_rows_;
	std::function<size_t(const std::pair<double, double>&)> xy_hash_;
};

} // namespace geom
#endif