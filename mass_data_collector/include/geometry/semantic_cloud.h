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

/* using classes instead of enum to work with std::enable_if */
namespace cloud_backend 
{
	class KD_TREE {};
	class SURFACE_MAP {};
}
enum class AggregationStrategy {
    HIGHEST_Z,
    MAJORITY,
	WEIGHTED_MAJORITY,
	HEURISTIC_1,
	HEURISTIC_2
};

// type traits
template <typename T>
struct uses_kd_tree {
	static constexpr bool value = false;
};

template <>
struct uses_kd_tree<cloud_backend::KD_TREE> {
	static constexpr bool value = true;
};

template <typename T>
struct uses_surface_map {
	static constexpr bool value = false;
};

template <>
struct uses_surface_map<cloud_backend::SURFACE_MAP> {
	static constexpr bool value = true;
};
#define KD_TREE_FUNC template <typename B = T, typename std::enable_if<uses_kd_tree<B>::value>::type* = nullptr>

template <typename T>
class SemanticCloud
{
public:
	using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor
														<double, SemanticCloud>,
														 SemanticCloud,
														 2>;
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
	explicit SemanticCloud(SemanticCloud::Settings settings);
	~SemanticCloud();
	// enabled for both specializations
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth);
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth,
							   Eigen::Matrix4d& transform);
	void AddFilteredSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   		   cv::Mat semantic,
							   		   cv::Mat depth);
	// enabled only for KD_TREE backend
	KD_TREE_FUNC
	std::tuple<cv::Mat, cv::Mat> GetSemanticBEV(double vehicle_width, double vehicle_length) const;
	cv::Mat GetFOVMask() const;
	KD_TREE_FUNC
	size_t GetMajorityVote(const std::vector<size_t>& knn_indices,
						   const std::vector<double>& distances) const;
	KD_TREE_FUNC
	std::pair<std::vector<size_t>, std::vector<double>> FindClosestPoints(double knn_x,
																		  double knn_y,
																		  size_t num_results) const;
	KD_TREE_FUNC
	void BuildKDTree();

	// enabled for both
	static cv::Mat ConvertToCityScapesPallete(cv::Mat semantic_ids);

	// deprecated functions
	KD_TREE_FUNC
	std::tuple<double, double, double, double> GetBoundaries() const;
	KD_TREE_FUNC
	void SaveCloud(const std::string& path) const;
	KD_TREE_FUNC
	std::tuple<double, double, double, double> GetVehicleBoundary() const;
	KD_TREE_FUNC
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


	SemanticCloud(const SemanticCloud&) = delete;
	const SemanticCloud& operator=(const SemanticCloud&) = delete;
	SemanticCloud(SemanticCloud&&) = delete;
	const SemanticCloud& operator=(SemanticCloud&&) = delete;
private:

	// members
	pcl::PointCloud<pcl::PointXYZL> target_cloud_;
	std::unique_ptr<KDTree2D> kd_tree_;
	std::unique_ptr<SurfaceMap<pcl::PointXYZL>> surface_map_;
	Settings cfg_;
	float pixel_w_;
	float pixel_h_;
	// std::function<size_t(const std::pair<double, double>&)> xy_hash_;
	// std::unordered_map<std::pair<double, double>, double, decltype(xy_hash_)> xy_map_;
};

} // namespace geom
#endif