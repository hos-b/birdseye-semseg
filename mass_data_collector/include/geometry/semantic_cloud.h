#ifndef __MASS_PCLOUD_H__
#define __MASS_PCLOUD_H__

#include <functional>
#include <memory>
#include <type_traits>

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>

#include "geometry/camera_geomtry.h"
#include "geometry/surface_map.h"

namespace geom 
{

/* ------------------------------------ semantic cloud class  ------------------------------------ */
class SemanticCloud
{
public:
	struct Settings {
		float max_point_x;
		float min_point_x;
		float max_point_y;
		float min_point_y;
		float stitching_threshold;
		unsigned int image_rows;
		unsigned int image_cols;
		unsigned int vehicle_mask_padding;
		unsigned int knn_count;
		unsigned int kd_max_leaf;
	};
	using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor
															<double, SemanticCloud>,
														 SemanticCloud,
														 2>;
	explicit SemanticCloud(Settings& settings);
	~SemanticCloud();
	SemanticCloud(const SemanticCloud&) = delete;
	const SemanticCloud& operator=(const SemanticCloud&) = delete;
	SemanticCloud(SemanticCloud&&) = delete;
	const SemanticCloud& operator=(SemanticCloud&&) = delete;


	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth);
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth,
							   Eigen::Matrix4d& transform);
	std::tuple<cv::Mat, cv::Mat> GetBEVData(double vehicle_width, double vehicle_length) const;
	cv::Mat GetFOVMask() const;
	size_t GetMajorityVote(const std::vector<size_t>& knn_indices,
						   const std::vector<double>& distances) const;

	void BuildKDTree();
	std::pair<std::vector<size_t>, std::vector<double>> FindClosestPoints(double knn_x,
																		  double knn_y,
																		  size_t num_results) const;
	// deprecated functions
	std::tuple<double, double, double, double> GetBoundaries() const;
	void SaveCloud(const std::string& path) const;
	std::tuple<double, double, double, double> GetVehicleBoundary() const;
	void SaveMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
						 const std::string& path, double pixel_limit) const;
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
	
	static cv::Mat ConvertToCityScapesPallete(cv::Mat semantic_ids);


private:
	Settings cfg_;
	float pixel_w_;
	float pixel_h_;
	pcl::PointCloud<pcl::PointXYZL> target_cloud_;
	std::unique_ptr<KDTree2D> kd_tree_;
};

} // namespace geom
#endif