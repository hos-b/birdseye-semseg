#include "geometry/semantic_cloud.h"
#include "config/agent_config.h"
#include "config/geom_config.h"

#include <cmath>
#include <tuple>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/io/pcd_io.h>

namespace geom
{
/* virtual base destructor of abstract class */
base_members::~base_members() {}

/* ------- KD-Tree template specialization ----------------------------------------------------- */
#pragma region KD_TREE
/* constructor */
template<>
SemanticCloud<CloudBackend::KD_TREE>::SemanticCloud(geom::base_members::Settings& settings) {
	cfg_ = settings;
	kd_tree_ = nullptr;
	pixel_w_ = (cfg_.max_point_y - cfg_.min_point_y) / static_cast<double>(cfg_.image_cols);
	pixel_h_ = (cfg_.max_point_x - cfg_.min_point_x) / static_cast<double>(cfg_.image_rows);
	target_cloud_.height = 1;
	target_cloud_.is_dense = true;
}

/* destructor: cleans containers */
template<>
SemanticCloud<CloudBackend::KD_TREE>::~SemanticCloud() {
	if (kd_tree_) {
		kd_tree_.reset();
		kd_tree_ = nullptr;
	}
	target_cloud_.points.clear();
}

/* converts a semantic|depth image into 3D points [in the car transform] and adds them to the point cloud  */
void cloud_base<CloudBackend::KD_TREE>::
		AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							  cv::Mat semantic,
							  cv::Mat depth) {
	size_t old_size = target_cloud_.points.size();
	target_cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
	// resizing omp critical|atomic is definitely than arbitrary emplace_back
	// but is at least accurate
	size_t index = old_size;
	# pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic.rows; ++i) {
		for (int j = 0; j < semantic.cols; ++j) {
			auto label = semantic.at<uchar>(i, j);
			// filter if it belongs to a traffic light or pole
			if (config::filtered_semantics[label]) {
				continue;
			}
			Eigen::Vector3d pixel_3d_loc = geometry->ReprojectToLocal(Eigen::Vector2d(i, j),
																	  depth.at<float>(i, j));
			// triggers a lot
			if (pixel_3d_loc.y() < cfg_.min_point_y || pixel_3d_loc.y() > cfg_.max_point_y ||
				pixel_3d_loc.x() < cfg_.min_point_x || pixel_3d_loc.x() > cfg_.max_point_x) {
				continue;
			}
			pcl::PointXYZL point;
			point.label = label;
			point.x = pixel_3d_loc.x();
			point.y = pixel_3d_loc.y();
			point.z = pixel_3d_loc.z();
			#pragma omp critical
			{
				target_cloud_.points[index] = point;
				index += 1;
			}
		}
	}
	target_cloud_.points.resize(index);
	target_cloud_.width = index;
}

/* simple kd-tree look up. only use whe the cloud is  */
std::pair<std::vector<size_t>, std::vector<double>> cloud_base<CloudBackend::KD_TREE>::
		FindClosestPoints(double knn_x, double knn_y, size_t num_results) const {
	double query[2] = {knn_x, knn_y};
	std::vector<size_t> knn_ret_index(num_results);
	std::vector<double> knn_sqrd_dist(num_results);
	num_results = kd_tree_->knnSearch(&query[0], num_results, &knn_ret_index[0], &knn_sqrd_dist[0]);
	// in case of less points in the tree than requested:
	knn_ret_index.resize(num_results);
	knn_sqrd_dist.resize(num_results);
	return std::make_pair(knn_ret_index, knn_sqrd_dist);
}

/* gets the majority vote for label based on the results of a knn search */
size_t cloud_base<CloudBackend::KD_TREE>::GetMajorityVote(const std::vector<size_t>& knn_indices,
					    								   const std::vector<double>& distances) const {
	// map from semantic id to count
	std::vector<double> class_weights(config::kCARLASemanticClassCount, 0.0);
	double max_weight = 0;
	size_t winner_semantic_id = 0;
	for (uint32_t i = 0; i < knn_indices.size(); ++i) {
		auto& current_point = target_cloud_.points[knn_indices[i]];
		unsigned int semantic_id = current_point.label;
		auto& class_weight = class_weights[semantic_id];
		class_weight += config::semantic_weight[semantic_id] + current_point.z * 100.0 +
						(1.0 / std::sqrt(distances[i]));
		if (class_weight > max_weight) {
			max_weight = class_weight;
			winner_semantic_id = semantic_id;
		}
	}
	return winner_semantic_id;
}

/* returns the orthographic bird's eye view image */
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::KD_TREE>::
		GetBEVData(double vehicle_width, double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Rect2d vhc_rect(cv::Point(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  			  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding),
						cv::Point(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   			  mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding));
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat vehicle_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			// get the center of the square that is to be mapped to a pixel
			double knn_x = cfg_.max_point_x - (i + 0.5) * pixel_h_;
			double knn_y = cfg_.max_point_y - (j + 0.5) * pixel_w_;
			// ------------------------ majority voting for semantic id ------------------------
			auto [knn_ret_index, knn_sqrd_dist] = FindClosestPoints(knn_x, knn_y, cfg_.knn_count);
			auto mv_label = GetMajorityVote(knn_ret_index, knn_sqrd_dist);
			semantic_bev.at<uchar>(i, j) = static_cast<uchar>(mv_label);
			// if the point belongs to the car itself
			if (mv_label == config::kCARLAVehiclesSemanticID &&
				vhc_rect.contains(cv::Point(j, i))) {
				vehicle_mask.at<uchar>(i, j) = 255;
				continue;
			}
		}
	}
	return std::make_tuple(semantic_bev, vehicle_mask);
}

/* returns the BEV mask visible from the camera. this method assumes the kd tree
   is populated only with points captured with the front camera */
cv::Mat cloud_base<CloudBackend::KD_TREE>::GetFOVMask() const {
	cv::Mat bev_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	# pragma omp parallel for collapse(2)
	for (int i = 0; i < bev_mask.rows; ++i) {
		// uchar* mask_row = bev_mask.ptr<uchar>(i);
		for (int j = 0; j < bev_mask.cols; ++j) {
			// get the center of the square that is to be mapped to a pixel
			double knn_x = cfg_.max_point_x - (i + 0.5) * pixel_h_;
			double knn_y = cfg_.max_point_y - (j + 0.5) * pixel_w_;
			auto [knn_ret_index, knn_sqrd_dist] = FindClosestPoints(knn_x, knn_y, cfg_.knn_count);
			// if the point is close enough
			bev_mask.at<uchar>(i, j) = static_cast<unsigned char>
				(knn_sqrd_dist[0] <= cfg_.stitching_threshold) * 255;
		}
	}
	return bev_mask;
}

/* builds an index out of the point cloud */
void cloud_base<CloudBackend::KD_TREE>::BuildKDTree() {
	kd_tree_ = std::make_unique<KDTree2D>(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(cfg_.kd_max_leaf));
	kd_tree_->buildIndex();
}

/* saves the color coded point cloud */
void cloud_base<CloudBackend::KD_TREE>::SaveCloud(const std::string& path) const {
	if (target_cloud_.empty()) {
		std::cout << "empty cloud, not saving." << std::endl;
		return;
	}
	pcl::PointCloud<pcl::PointXYZRGB> rgb_cloud;
	rgb_cloud.points.resize(target_cloud_.points.size());
	size_t size = 0;
	cv::Scalar color;
	// #pragma omp parallel for
	for (size_t i = 0; i < target_cloud_.points.size(); ++i) {
		rgb_cloud.points[size].x = target_cloud_.points[i].x;
		rgb_cloud.points[size].y = target_cloud_.points[i].y;
		rgb_cloud.points[size].z = target_cloud_.points[i].z;
		color = config::carla_to_cityscapes_palette[target_cloud_.points[i].label];
		rgb_cloud.points[size].b = color[0];
		rgb_cloud.points[size].g = color[1];
		rgb_cloud.points[size].r = color[2];
		++size;
	}
	rgb_cloud.width = size;
	rgb_cloud.height = 1;
	rgb_cloud.is_dense = true;
	pcl::io::savePCDFile(path, rgb_cloud);
}


/* returns the visible points in the cloud that are visible in the given camera geometry */
void cloud_base<CloudBackend::KD_TREE>::
	SaveMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
					const std::string& path, double pixel_limit) const {
	pcl::PointCloud<pcl::PointXYZRGB> visible_cloud;
	visible_cloud.points.resize(target_cloud_.points.size());
	size_t size = 0;
	cv::Scalar color;
	#pragma omp parallel for
	for (auto it = target_cloud_.points.begin(); it != target_cloud_.points.end(); it += 1) {
		if (rgb_geometry->IsInView(*it, pixel_limit)) {
			visible_cloud.points[size].x = it->x;
			visible_cloud.points[size].y = it->y;
			visible_cloud.points[size].z = it->z;
			color = config::carla_to_cityscapes_palette[it->label];
			visible_cloud.points[size].b = color[0];
			visible_cloud.points[size].g = color[1];
			visible_cloud.points[size].r = color[2];
			++size;
		}
	}
	visible_cloud.points.resize(size);
	visible_cloud.width = size;
	visible_cloud.height = 1;
	visible_cloud.is_dense = true;
	pcl::io::savePCDFile(path, visible_cloud);
}

#pragma endregion
/* ------- SurfaceMap template specialization -------------------------------------------------- */
#pragma region SURFACE_MAP
/* constructor */
template<>
SemanticCloud<CloudBackend::SURFACE_MAP>::SemanticCloud(geom::base_members::Settings& settings) {
	cfg_ = settings;
	pixel_w_ = (cfg_.max_point_y - cfg_.min_point_y) / static_cast<double>(cfg_.image_cols);
	pixel_h_ = (cfg_.max_point_x - cfg_.min_point_x) / static_cast<double>(cfg_.image_rows);
	SurfaceMapSettings smap_settings{pixel_h_, pixel_w_,
									 cfg_.max_point_x, cfg_.max_point_y,
									 cfg_.image_rows , cfg_.image_cols};
	surface_map_ = std::make_unique<SurfaceMap<pcl::PointXYZL, OverflowBehavior::IGNORE, 200>>(smap_settings);
}

/* destructor: cleans containers */
template<>
SemanticCloud<CloudBackend::SURFACE_MAP>::~SemanticCloud() {
	surface_map_.reset();
}

/* converts a semantic|depth image into 3D points [in egocar transform] and adds them to the point cloud  */
void cloud_base<CloudBackend::SURFACE_MAP>::
		AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							  cv::Mat semantic,
							  cv::Mat depth) {
	// resizing + omp critical|atomic is definitely worse
	# pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic.rows; ++i) {
		for (int j = 0; j < semantic.cols; ++j) {
			auto label = semantic.at<uchar>(i, j);
			// filter if it belongs to a traffic light or pole
			if (config::filtered_semantics[label]) {
				continue;
			}
			Eigen::Vector3d pixel_3d_loc =
				geometry->ReprojectToLocal(Eigen::Vector2d(i, j), depth.at<float>(i, j));
			pcl::PointXYZL point;
			point.label = label;
			point.x = pixel_3d_loc.x();
			point.y = pixel_3d_loc.y();
			point.z = pixel_3d_loc.z();
			// #pragma omp critical
			surface_map_->AddPoint(point);
		}
	}
}

/* returns the orthographic bird's eye view image using highest z strategy */
template<>
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::SURFACE_MAP>::
		GetBEVData<AggregationStrategy::HIGHEST_Z>(double vehicle_width, double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding);
	cv::Rect2d ego_vehicle_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat vehicle_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			auto& points = surface_map_->GetGridPoints(j, i);
			// skip if bin is empty, shouldn't happen
			if (points.size() == 0) {
				continue;
			}
			pcl::PointXYZL strat_winner = points.front();
			for (auto it = points.begin(); it != points.end(); it += 1) {
				if (it->z > strat_winner.z) {
					strat_winner = *it;
				}
			}
			semantic_bev.at<uchar>(i, j) = static_cast<uchar>(strat_winner.label);
			// if the point belongs to the car itself
			if (strat_winner.label == config::kCARLAVehiclesSemanticID &&
					ego_vehicle_rect.contains(cv::Point(j, i))) {
				vehicle_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	return std::make_tuple(semantic_bev, vehicle_mask);
}

/* returns the orthographic bird's eye view image using majority voting strategy */
template<>
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::SURFACE_MAP>::
		GetBEVData<AggregationStrategy::MAJORITY>(double vehicle_width, double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding);
	cv::Rect2d ego_vehicle_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat vehicle_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			auto& points = surface_map_->GetGridPoints(j, i);
			// skip if bin is empty
			if (points.size() == 0) {
				// TODO: figure out not skipping
				continue;
			}
			std::vector<size_t> class_counts(config::kCARLASemanticClassCount, 0);
			uint8_t winner_class = 0;
			size_t winner_count = 0;
			for (auto& point : points) {
				auto& cls_count = class_counts[point.label];
				cls_count += 1;
				if (cls_count > winner_count) {
					winner_count = cls_count;
					winner_class = static_cast<uint8_t>(point.label);
				}
			}
			semantic_bev.at<uchar>(i, j) = winner_class;
			// if the point belongs to the car itself
			if (winner_class == config::kCARLAVehiclesSemanticID &&
					ego_vehicle_rect.contains(cv::Point(j, i))) {
				vehicle_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	return std::make_tuple(semantic_bev, vehicle_mask);
}

/* returns the orthographic bird's eye view image using weighted majority voting strategy */
template<>
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::SURFACE_MAP>::
		GetBEVData<AggregationStrategy::WEIGHTED_MAJORITY>(double vehicle_width, double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding);
	cv::Rect2d ego_vehicle_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat vehicle_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			auto& points = surface_map_->GetGridPoints(j, i);
			// skip if bin is empty
			if (points.size() == 0) {
				continue;
			}
			std::vector<double> class_weights(config::kCARLASemanticClassCount, 0.0);
			size_t winner_class = 0;
			double winner_weight = 0;
			for (auto& point : points) {
				auto label = point.label;
				auto& cls_weight = class_weights[label];
				cls_weight += config::semantic_weight[label] + point.z * 100.0;
				if (cls_weight > winner_weight) {
					winner_weight = cls_weight;
					winner_class = label;
				}
			}
			semantic_bev.at<uchar>(i, j) = static_cast<uchar>(winner_class);
			// if the point belongs to the car itself
			if (winner_class == config::kCARLAVehiclesSemanticID &&
					ego_vehicle_rect.contains(cv::Point(j, i))) {
				vehicle_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	return std::make_tuple(semantic_bev, vehicle_mask);
}

/* returns the BEV mask visible from the camera. this method assumes the kd tree
   is populated only with points captured with the front camera */
cv::Mat cloud_base<CloudBackend::SURFACE_MAP>::GetFOVMask(size_t min_point_count) const {
	cv::Mat bev_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	# pragma omp parallel for collapse(2)
	for (int i = 0; i < bev_mask.rows; ++i) {
		for (int j = 0; j < bev_mask.cols; ++j) {
			if (surface_map_->GetGridPoints(j, i).size() >= min_point_count) {
				bev_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	return bev_mask;
}
#pragma endregion
/* ------- Mixed template specialization -------------------------------------------- */
#pragma region MIXED
/* constructor */
template<>
SemanticCloud<CloudBackend::MIXED>::SemanticCloud(geom::base_members::Settings& settings) {
	cfg_ = settings;
	pixel_w_ = (cfg_.max_point_y - cfg_.min_point_y) / static_cast<double>(cfg_.image_cols);
	pixel_h_ = (cfg_.max_point_x - cfg_.min_point_x) / static_cast<double>(cfg_.image_rows);
	SurfaceMapSettings smap_settings{pixel_h_, pixel_w_,
									 cfg_.max_point_x, cfg_.max_point_y,
									 cfg_.image_rows , cfg_.image_cols};
	semantic_surface_map_ =
		std::make_unique<SurfaceMap<pcl::PointXYZL, OverflowBehavior::IGNORE, 200>>(smap_settings);
	mask_kd_tree_ = nullptr;
	mask_cloud_.height = 1;
	mask_cloud_.is_dense = true;
}

/* destructor: cleans containers */
template<>
SemanticCloud<CloudBackend::MIXED>::~SemanticCloud() {
	semantic_surface_map_.reset();
	if (mask_kd_tree_) {
		mask_kd_tree_.reset();
		mask_kd_tree_ = nullptr;
	}
	mask_cloud_.points.clear();
}

/* builds an index out of the point cloud */
void cloud_base<CloudBackend::MIXED>::BuildKDTree() {
	mask_kd_tree_ = std::make_unique<KDTree2D>(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(cfg_.kd_max_leaf));
	mask_kd_tree_->buildIndex();
}

/* converts a semantic|depth image into 3D points [in egocar transform] and adds them to the point cloud 
   for this specific template specialization (backend::MIXED, dest::MASK) it should only be called once.
*/
template <>
void cloud_base<CloudBackend::MIXED>::
		AddSemanticDepthImage<DestinationMap::MASK>(std::shared_ptr<geom::CameraGeometry> geometry,
							  						cv::Mat semantic,
							  						cv::Mat depth) {
	mask_cloud_.points.resize(semantic.rows * semantic.cols);
	// resizing omp critical|atomic is definitely than arbitrary emplace_back
	// but is at least accurate
	size_t index = 0;
	// resizing + omp critical|atomic is definitely worse but more accurate
	# pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic.rows; ++i) {
		for (int j = 0; j < semantic.cols; ++j) {
			auto label = semantic.at<uchar>(i, j);
			// filter if it belongs to a traffic light or pole
			if (config::filtered_semantics[label]) {
				continue;
			}
			Eigen::Vector3d pixel_3d_loc =
				geometry->ReprojectToLocal(Eigen::Vector2d(i, j), depth.at<float>(i, j));
			// triggers a lot
			if (pixel_3d_loc.y() < cfg_.min_point_y || pixel_3d_loc.y() > cfg_.max_point_y ||
				pixel_3d_loc.x() < cfg_.min_point_x || pixel_3d_loc.x() > cfg_.max_point_x) {
				continue;
			}
			pcl::PointXYZL point;
			point.label = label;
			point.x = pixel_3d_loc.x();
			point.y = pixel_3d_loc.y();
			point.z = pixel_3d_loc.z();
			#pragma omp critical
			{
				mask_cloud_.points[index] = point;
				index += 1;
			}
		}
	}
	mask_cloud_.points.resize(index);
	BuildKDTree();
}

/* performs a kd-tree lookup and returns 255 if the closest point is in the defined threshold, 
   0 otherwise
*/
uint8_t cloud_base<CloudBackend::MIXED>::
		GetMaskKDValue(size_t row, size_t col, size_t num_results) const {
	double query[2] = {cfg_.max_point_x - (row + 0.5) * pixel_h_,
					   cfg_.max_point_y - (col + 0.5) * pixel_w_};
	std::vector<size_t> knn_ret_index(num_results);
	std::vector<double> knn_sqrd_dist(num_results);
	num_results = mask_kd_tree_->knnSearch(&query[0], num_results, &knn_ret_index[0], &knn_sqrd_dist[0]);
	// in case of less points in the tree than requested:
	knn_ret_index.resize(num_results);
	knn_sqrd_dist.resize(num_results);
	return static_cast<unsigned char>(knn_sqrd_dist[0] <= cfg_.stitching_threshold) * 255;
}

/* converts a semantic|depth image into 3D points [in egocar transform] and adds them to the point cloud  */
template <>
void cloud_base<CloudBackend::MIXED>::
		AddSemanticDepthImage<DestinationMap::SEMANTIC>(std::shared_ptr<geom::CameraGeometry> geometry,
							  							cv::Mat semantic,
							  							cv::Mat depth) {
	// resizing + omp critical|atomic is definitely worse
	# pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic.rows; ++i) {
		for (int j = 0; j < semantic.cols; ++j) {
			auto label = semantic.at<uchar>(i, j);
			// filter if it belongs to a traffic light or pole
			if (config::filtered_semantics[label]) {
				continue;
			}
			Eigen::Vector3d pixel_3d_loc =
				geometry->ReprojectToLocal(Eigen::Vector2d(i, j), depth.at<float>(i, j));
			pcl::PointXYZL point;
			point.label = label;
			point.x = pixel_3d_loc.x();
			point.y = pixel_3d_loc.y();
			point.z = pixel_3d_loc.z();
			#pragma omp critical
			{
			semantic_surface_map_->AddPoint(point);
			}
		}
	}
}

/* returns the orthographic bird's eye view image using highest z strategy */
template<>
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::MIXED>::
		GetBEVData<AggregationStrategy::HIGHEST_Z>(double vehicle_width,
												   double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding);
	cv::Rect2d ego_vehicle_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			// semantic point container
			auto& semantic_points = semantic_surface_map_->GetGridPoints(j, i);
			pcl::PointXYZL strat_winner = semantic_points.front();
			for (auto it = semantic_points.begin(); it != semantic_points.end(); it += 1) {
				if (it->z > strat_winner.z) {
					strat_winner = *it;
				}
			}
			semantic_bev.at<uchar>(i, j) = static_cast<uint8_t>(strat_winner.label);
			// if the point belongs to the car itself
			if (strat_winner.label == config::kCARLAVehiclesSemanticID &&
					ego_vehicle_rect.contains(cv::Point(j, i))) {
				mask.at<uchar>(i, j) = 255;
				continue;
			}
			// fill out the rest of the mask using a kd-tree lookup
			mask.at<uchar>(i, j) = GetMaskKDValue(i, j, cfg_.knn_count);
		}
	}
	return std::make_tuple(semantic_bev, mask);
}

/* returns the orthographic bird's eye view image using majority voting strategy */
template<>
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::MIXED>::
		GetBEVData<AggregationStrategy::MAJORITY>(double vehicle_width,
												  double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding);
	cv::Rect2d ego_vehicle_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			auto& semantic_points = semantic_surface_map_->GetGridPoints(j, i);
			std::vector<size_t> class_counts(config::kCARLASemanticClassCount, 0);
			uint8_t winner_class = 0;
			size_t winner_count = 0;
			for (auto& point : semantic_points) {
				auto& cls_count = class_counts[point.label];
				cls_count += 1;
				if (cls_count > winner_count) {
					winner_count = cls_count;
					winner_class = static_cast<uint8_t>(point.label);
				}
			}
			semantic_bev.at<uchar>(i, j) = winner_class;
			// if the point belongs to the car itself
			if (winner_class == config::kCARLAVehiclesSemanticID &&
					ego_vehicle_rect.contains(cv::Point(j, i))) {
				mask.at<uchar>(i, j) = 255;
				continue;
			}
			// fill out the rest of the mask using a kd-tree lookup
			mask.at<uchar>(i, j) = GetMaskKDValue(i, j, cfg_.knn_count);
		}
	}
	return std::make_tuple(semantic_bev, mask);
}

/* returns the orthographic bird's eye view image using weighted majority voting strategy */
template<>
std::tuple<cv::Mat, cv::Mat> cloud_base<CloudBackend::MIXED>::
		GetBEVData<AggregationStrategy::WEIGHTED_MAJORITY>(double vehicle_width,
														   double vehicle_length) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - cfg_.vehicle_mask_padding,
					  mid.y - (vehicle_length / (2 * pixel_h_)) - cfg_.vehicle_mask_padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + cfg_.vehicle_mask_padding,
					   mid.y + (vehicle_length / (2 * pixel_h_)) + cfg_.vehicle_mask_padding);
	cv::Rect2d ego_vehicle_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			auto& semantic_points = semantic_surface_map_->GetGridPoints(j, i);
			std::vector<double> class_weights(config::kCARLASemanticClassCount, 0.0);
			uint8_t winner_class = 0;
			double winner_weight = 0;
			for (auto& point : semantic_points) {
				auto label = point.label;
				auto& cls_weight = class_weights[label];
				cls_weight += config::semantic_weight[label] + point.z * 100.0;
				if (cls_weight > winner_weight) {
					winner_weight = cls_weight;
					winner_class = static_cast<uint8_t>(label);
				}
			}
			semantic_bev.at<uchar>(i, j) = winner_class;
			// if the point belongs to the car itself
			if (winner_class == config::kCARLAVehiclesSemanticID &&
					ego_vehicle_rect.contains(cv::Point(j, i))) {
				mask.at<uchar>(i, j) = 255;
				continue;
			}
			// fill out the rest of the mask using a kd-tree lookup
			mask.at<uchar>(i, j) = GetMaskKDValue(i, j, cfg_.knn_count);
		}
	}
	return std::make_tuple(semantic_bev, mask);
}
#pragma endregion
/* --------- Shared definitions ---------------------------------------------------------------- */

/* converts the smenatic mat to an RGB image */
template <CloudBackend B>
cv::Mat SemanticCloud<B>::ConvertToCityScapesPallete(cv::Mat semantic_ids) {
	cv::Mat rgb_mat(semantic_ids.rows, semantic_ids.cols, CV_8UC3);
	#pragma omp parallel for
	for (int i = 0; i < semantic_ids.rows; ++i) {
		cv::Vec3b* rgb_row = rgb_mat.ptr<cv::Vec3b>(i);
		uchar* semantic_row = semantic_ids.ptr<uchar>(i);
		for (int j = 0; j < semantic_ids.cols; ++j) {
			rgb_row[j] = config::carla_to_cityscapes_palette[semantic_row[j]];
		}
	}
	return rgb_mat;
}

/* ------- Deprecated functions ---------------------------------------------------------------- */
#pragma region deprecated
/* returns the boundaries of the car (calculated from filtered point cloud) */
[[deprecated("deprecated function")]]
std::tuple<double, double, double, double>
		cloud_base<CloudBackend::KD_TREE>::GetVehicleBoundary() const {
	double front = 0.0;
	double back = 0.0;
	double left = 0.0;
	double right = 0.0;
	double seek = 0.0;
	// going cm by cm along the axes to find the boundaries
	while (true) {
		seek += 0.01;
		auto [pt_indices, pt_dists] = FindClosestPoints(seek, 0, 1);
		const auto& next_point = target_cloud_.points[pt_indices[0]];
		if (next_point.label == config::kCARLAVehiclesSemanticID) {
			front = next_point.x;
		} else {
			break;
		}
	}
	seek = 0;
	while (true) {
		seek -= 0.01;
		auto [pt_indices, pt_dists] = FindClosestPoints(seek, 0, 1);
		const auto& next_point = target_cloud_.points[pt_indices[0]];
		if (next_point.label == config::kCARLAVehiclesSemanticID) {
			back = next_point.x;
		} else {
			break;
		}
	}
	seek = 0;
	while (true) {
		seek += 0.01;
		auto [pt_indices, pt_dists] = FindClosestPoints(0, seek, 1);
		const auto& next_point = target_cloud_.points[pt_indices[0]];
		if (next_point.label == config::kCARLAVehiclesSemanticID) {
			left = next_point.y;
		} else {
			break;
		}
	}
	seek = 0;
	while (true) {
		seek -= 0.01;
		auto [pt_indices, pt_dists] = FindClosestPoints(0, seek, 1);
		const auto& next_point = target_cloud_.points[pt_indices[0]];
		if (next_point.label == config::kCARLAVehiclesSemanticID) {
			right = next_point.y;
		} else {
			break;
		}
	}
	return std::make_tuple(front, std::abs(back), left, std::abs(right));
}

/* debug function that returns the x,y boundaries of the cloud */
std::tuple<double, double, double, double>
		cloud_base<CloudBackend::KD_TREE>::GetBoundaries() const {
	float minx = 0, miny = 0, maxx = 0, maxy = 0;
	for (const auto& point : target_cloud_.points) {
		if (point.x < minx) {
			minx = point.x;
		} else if (point.x > maxx) {
			maxx = point.x;
		}
		if (point.y < miny) {
			miny = point.y;
		} else if (point.y > maxy) {
			maxy = point.y;
		}
	}
	return std::make_tuple(minx, maxx, miny, maxy);
}

/* converts a semantic|depth image into 3D points [in the global transform] and adds them to the point cloud  */
void cloud_base<CloudBackend::KD_TREE>::
		AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
							  cv::Mat semantic,
							  cv::Mat depth,
							  Eigen::Matrix4d& transform) {
	size_t old_size = target_cloud_.points.size();
	target_cloud_.points.reserve(old_size + (semantic.rows * semantic.cols));
	// # pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic.rows; ++i) {
		uchar* pixel_label = semantic.ptr<uchar>(i);
		float* pixel_depth = depth.ptr<float>(i);
		for (int j = 0; j < semantic.cols; ++j) {
			Eigen::Vector3d pixel_3d_loc = geometry->ReprojectToGlobal(Eigen::Vector2d(i, j), transform, pixel_depth[j]);
			pcl::PointXYZL point;
			point.label = pixel_label[j];
			point.x = pixel_3d_loc.x();
			point.y = pixel_3d_loc.y();
			point.z = pixel_3d_loc.z();
			target_cloud_.points.emplace_back(point);
		}
	}
	target_cloud_.width = target_cloud_.points.size();
}
#pragma endregion
} // namespace geom