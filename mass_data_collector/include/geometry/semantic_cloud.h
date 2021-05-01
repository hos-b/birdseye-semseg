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

/* enum for choosing the backend point cloud container of SemanticCloud */
enum class CloudBackend  {
	KD_TREE,
	SURFACE_MAP,
	DOUBLE_SURFACE_MAP
};
/* enum for aggregation strategy in surface map backends */
enum class AggregationStrategy {
    HIGHEST_Z,
    MAJORITY,
	WEIGHTED_MAJORITY,
};
/* enum for choosing the destination container in double surface map backend */
enum class DestinationMap {
	MASK,
	SEMANTIC
};
/* ----------- cloud_base & base_members abstract class  ---------------------------------- */
/* base members inherited by all cloud backends */
class base_members {
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
	// virtual destructor to make the class abstract
	virtual ~base_members() = 0;
protected:
	Settings cfg_;
	float pixel_w_;
	float pixel_h_;
};
/* cloud_base class */
template <CloudBackend B>
class cloud_base : protected base_members {};
/* ----------- cloud_base kd tree specialization  ----------------------------------------- */
template <>
class cloud_base<CloudBackend::KD_TREE> : protected base_members {
public:
	using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor
															<double, cloud_base>,
															cloud_base,
															2>;
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
	std::pair<std::vector<size_t>, std::vector<double>> FindClosestPoints(double knn_x,
																		  double knn_y,
																		  size_t num_results) const;
	void BuildKDTree();
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

protected:
	pcl::PointCloud<pcl::PointXYZL> target_cloud_;
	std::unique_ptr<KDTree2D> kd_tree_;
};

/* ----------- cloud_base surface map specialization  ------------------------------------- */
template <>
class cloud_base<CloudBackend::SURFACE_MAP> : protected base_members {
public:
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth);
	cv::Mat GetFOVMask(size_t min_point_count) const;
	template <AggregationStrategy S>
	std::tuple<cv::Mat, cv::Mat> GetBEVData(double vehicle_width, double vehicle_length) const;

protected:
	std::unique_ptr<SurfaceMap<pcl::PointXYZL, OverflowBehavior::IGNORE, 200>> surface_map_;
};
/* ----------- cloud_base double surface map specialization  ------------------------------ */
template <>
class cloud_base<CloudBackend::DOUBLE_SURFACE_MAP> : protected base_members {
public:
	template <DestinationMap M>
	void AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							   cv::Mat semantic,
							   cv::Mat depth);
	template <AggregationStrategy S>
	std::tuple<cv::Mat, cv::Mat> GetBEVData(double vehicle_width,
											double vehicle_length,
											size_t min_point_count) const;
protected:
	std::unique_ptr<SurfaceMap<pcl::PointXYZL, OverflowBehavior::IGNORE, 200>> mask_surface_map_;
	std::unique_ptr<SurfaceMap<pcl::PointXYZL, OverflowBehavior::IGNORE, 200>> semantic_surface_map_;
};
/* ----------- templated semantic cloud class  -------------------------------------------- */
template <CloudBackend B>
class SemanticCloud : public cloud_base<B>
{
public:
	explicit SemanticCloud(geom::base_members::Settings& settings);
	~SemanticCloud();
	
	static cv::Mat ConvertToCityScapesPallete(cv::Mat semantic_ids);

	SemanticCloud(const SemanticCloud&) = delete;
	const SemanticCloud& operator=(const SemanticCloud&) = delete;
	SemanticCloud(SemanticCloud&&) = delete;
	const SemanticCloud& operator=(SemanticCloud&&) = delete;
};

} // namespace geom
#endif