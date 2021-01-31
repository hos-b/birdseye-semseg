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

class SemanticCloud
{
using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, SemanticCloud>,
													 SemanticCloud, 2>;
public:
	struct Settings {
		float max_point_x;
		float min_point_x;
		float max_point_y;
		float min_point_y;
		float image_rows;
		float image_cols;
		float sitching_threhsold;
		float vehicle_mask_padding;
		unsigned int knn_count;
	};
	SemanticCloud(SemanticCloud::Settings settings);
	~SemanticCloud();
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth);
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth,
							   Eigen::Matrix4d& transform);
	std::tuple<cv::Mat, cv::Mat> GetSemanticBEV(size_t knn_pt_count, double vehicle_width,
												double vehicle_length, size_t padding) const;
	cv::Mat GetFOVMask(double stitching_threshold) const;
	void BuildKDTree();
	static cv::Mat ConvertToCityScapesPallete(cv::Mat semantic_ids);

	// debug functions
	std::tuple<double, double, double, double> GetBoundaries() const;
	void SaveCloud(const std::string& path) const;
	std::tuple<double, double, double, double> GetVehicleBoundary() const;
	void SaveMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
						 const std::string& path, double pixel_limit);

	// mandatory kd-tree stuff
	inline size_t kdtree_get_point_count() const { return target_cloud_.points.size(); }
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
		if (dim == 0) {
			return target_cloud_.points[idx].x;
		}
		return target_cloud_.points[idx].y;
	}
	template<class BBox>
	bool kdtree_get_bbox(BBox& /* bb */) const { return false; }

private:
	size_t GetMajorityVote(const std::vector<size_t>& knn_indices,
						   const std::vector<double>& distances) const;
	std::pair<std::vector<size_t>, std::vector<double>> FindClosestPoints(double knn_x,
																		  double knn_y,
																		  size_t num_results) const;

	// members
	std::unique_ptr<KDTree2D> kd_tree_;
	pcl::PointCloud<pcl::PointXYZL> target_cloud_;
	Settings cfg_;
	double pixel_w_;
	double pixel_h_;
	std::function<size_t(const std::pair<double, double>&)> xy_hash_;
	std::unordered_map<std::pair<double, double>, double, decltype(xy_hash_)> xy_map_;
};

} // namespace geom
#endif