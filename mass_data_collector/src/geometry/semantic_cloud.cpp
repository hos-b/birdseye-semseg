#include "geometry/semantic_cloud.h"
#include "boost/algorithm/string/erase.hpp"
#include "config/agent_config.h"
#include "config/geom_config.h"

#include <bits/stdint-uintn.h>
#include <cmath>
#include <functional>
#include <deque>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>
#include <queue>
#include <pcl/io/pcd_io.h>

namespace geom
{
/* constructor */
SemanticCloud::SemanticCloud(SemanticCloud::Settings settings) {
	cfg_ = settings;
	kd_tree_ = nullptr;
	xy_hash_ = [](const std::pair<double, double>& pair) -> size_t {
		return std::hash<double>()(pair.first * 1e4) ^ std::hash<double>()(pair.second);
	};
	pixel_w_ = (cfg_.max_point_y - cfg_.min_point_y) / static_cast<double>(cfg_.image_cols);
	pixel_h_ = (cfg_.max_point_x - cfg_.min_point_x) / static_cast<double>(cfg_.image_rows);
	xy_map_ = std::unordered_map<std::pair<double, double>, double, decltype(xy_hash_)>(150000, xy_hash_);
}

/* destructor: cleans containers */
SemanticCloud::~SemanticCloud() {
	if (kd_tree_) {
		kd_tree_.reset();
		kd_tree_ = nullptr;
	}
	target_cloud_.points.clear();
}

/* converts a semantic|depth image into 3D points [in the car transform] and adds them to the point cloud  */
void SemanticCloud::AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
										  cv::Mat semantic,
										  cv::Mat depth) {
	size_t old_size = target_cloud_.points.size();
	target_cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
	size_t index;
	#pragma omp parallel for collapse(2) private(index) shared(target_cloud_, old_size)
	for (int i = 0; i < semantic.rows; ++i) {
		// uchar* pixel_label = semantic.ptr<uchar>(i);
		// float* pixel_depth = depth.ptr<float>(i);
		for (int j = 0; j < semantic.cols; ++j) {
			auto label = semantic.at<uchar>(i, j);
			// filter if it belongs to a traffic light or pole
			if (config::fileterd_semantics.at(label)) {
				continue;
			}
			Eigen::Vector3d pixel_3d_loc = geometry->ReprojectToLocal(Eigen::Vector2d(i, j), depth.at<float>(i, j));
			// check if the point is outside the specified pointcloud dimensions
			/* 
			if (pixel_3d_loc.y() < cfg_.min_point_y || pixel_3d_loc.y() > cfg_.max_point_y ||
				pixel_3d_loc.x() < cfg_.min_point_x || pixel_3d_loc.x() > cfg_.max_point_x) {
				continue;
			}
			// skip if repetitive or lower height, surprisingly this never triggers
			auto xy_pair = std::make_pair(pixel_3d_loc.x(), pixel_3d_loc.y());
			auto it = xy_map_.find(xy_pair);
			if (it != xy_map_.end() && it->second > pixel_3d_loc.z()) {
				continue;
			}
			xy_map_[xy_pair] = pixel_3d_loc.z(); */
			// private omp variable
			index = old_size + semantic.rows * i + j;
			target_cloud_[index].label = label;
			target_cloud_[index].x = pixel_3d_loc.x();
			target_cloud_[index].y = pixel_3d_loc.y();
			target_cloud_[index].z = pixel_3d_loc.z();
		}
	}
	// target_cloud_.points.resize(index);
	target_cloud_.width = target_cloud_.points.size();
	target_cloud_.height = 1;
	target_cloud_.is_dense = true;
}

/* converts a semantic|depth image into 3D points [in the global transform] and adds them to the point cloud  */
void SemanticCloud::AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
										  cv::Mat semantic,
										  cv::Mat depth,
										  Eigen::Matrix4d& transform) {
	size_t old_size = target_cloud_.points.size();
	size_t index = old_size;
	target_cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
	for (int i = 0; i < semantic.rows; ++i) {
		uchar* pixel_label = semantic.ptr<uchar>(i);
		float* pixel_depth = depth.ptr<float>(i);
		for (int j = 0; j < semantic.cols; ++j) {
			Eigen::Vector3d pixel_3d_loc = geometry->ReprojectToGlobal(Eigen::Vector2d(i, j), transform, pixel_depth[j]);
			target_cloud_[index].label = pixel_label[j];
			target_cloud_[index].x = pixel_3d_loc.x();
			target_cloud_[index].y = pixel_3d_loc.y();
			target_cloud_[index].z = pixel_3d_loc.z();
			++index;
		}
	}
	target_cloud_.points.resize(index);
	target_cloud_.width = index;
	target_cloud_.height = 1;
	target_cloud_.is_dense = true;
}

/* removes overlapping or invisible points for the target cloud. initializes kd tree.
   filters unwanted labels (declared in config::filtered_semantics)
*/
void SemanticCloud::BuildKDTree() {
	// building kd-tree
	kd_tree_ = std::make_unique<KDTree2D>(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(128));
	kd_tree_->buildIndex();
}

/* returns the orthographic bird's eye view image */
std::tuple<cv::Mat, cv::Mat> SemanticCloud::GetSemanticBEV(size_t knn_pt_count, double vehicle_width, double vehicle_length, size_t padding) const {
	// ego roi
	auto ego_row_px = (cfg_.max_point_x / (cfg_.max_point_x - cfg_.min_point_x)) * cfg_.image_rows;
	auto ego_col_px = (cfg_.max_point_y / (cfg_.max_point_y - cfg_.min_point_y)) * cfg_.image_cols;
	cv::Point mid(ego_col_px, ego_row_px);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)) - padding, mid.y - (vehicle_length / (2 * pixel_h_)) - padding);
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)) + padding, mid.y + (vehicle_length / (2 * pixel_h_)) + padding);
	cv::Rect2d vhc_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	cv::Mat vehicle_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < semantic_bev.rows; ++i) {
		// uchar* semantic_row = semantic_bev.ptr<uchar>(i);
		// uchar* mask_row = vehicle_mask.ptr<uchar>(i);
		for (int j = 0; j < semantic_bev.cols; ++j) {
			// get the center of the square that is to be mapped to a pixel
			double knn_x = cfg_.max_point_x - i * pixel_h_ + 0.5 * pixel_h_;
			double knn_y = cfg_.max_point_y - j * pixel_w_ + 0.5 * pixel_w_;
			// ------------------------ majority voting for semantic id ------------------------
			auto [knn_ret_index, knn_sqrd_dist] = FindClosestPoints(knn_x, knn_y, knn_pt_count);
			auto mv_winner = target_cloud_.points[GetMajorityVote(knn_ret_index, knn_sqrd_dist)];
			semantic_bev.at<uchar>(i, j) = static_cast<unsigned char>(mv_winner.label);
			// if the point belongs to the car itself
			if (mv_winner.label == config::kCARLAVehiclesSemanticID &&
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
cv::Mat SemanticCloud::GetFOVMask(double stitching_threshold) const {
	cv::Mat bev_mask = cv::Mat::zeros(cfg_.image_rows, cfg_.image_cols, CV_8UC1);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < bev_mask.rows; ++i) {
		// uchar* mask_row = bev_mask.ptr<uchar>(i);
		for (int j = 0; j < bev_mask.cols; ++j) {
			// get the center of the square that is to be mapped to a pixel
			double knn_x = cfg_.max_point_x - i * pixel_h_ + 0.5 * pixel_h_;
			double knn_y = cfg_.max_point_y - j * pixel_w_ + 0.5 * pixel_w_;
			auto [knn_ret_index, knn_sqrd_dist] = FindClosestPoints(knn_x, knn_y, 32);
			// if the point is close enough
			if (knn_sqrd_dist[0] <= stitching_threshold) {
				bev_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	return bev_mask;
}

/* simple kd-tree look up. only use whe the cloud is  */
std::pair<std::vector<size_t>, std::vector<double>>
SemanticCloud::FindClosestPoints(double knn_x, double knn_y, size_t num_results) const {
	double query[2] = {knn_x, knn_y};
	std::vector<size_t> knn_ret_index(num_results);
	std::vector<double> knn_sqrd_dist(num_results);
	num_results = kd_tree_->knnSearch(&query[0], num_results, &knn_ret_index[0], &knn_sqrd_dist[0]);
	// in case of less points in the tree than requested:
	knn_ret_index.resize(num_results);
	knn_sqrd_dist.resize(num_results);
	return std::make_pair(knn_ret_index, knn_sqrd_dist);
}

/* returns the boundaries of the car (calculated from filtered point cloud) */
[[deprecated("deprecated function")]] std::tuple<double, double, double, double>
SemanticCloud::GetVehicleBoundary() const {
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

/* returns the visible points in the cloud that are visible in the given camera geometry */
void SemanticCloud::SaveMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
									const std::string& path, double pixel_limit) {
	pcl::PointCloud<pcl::PointXYZRGB> visible_cloud;
	visible_cloud.points.resize(target_cloud_.points.size());
	size_t size = 0;
	cv::Scalar color;
	#pragma omp parallel for
	for (size_t i = 0; i < target_cloud_.points.size(); ++i) {
		if (rgb_geometry->IsInView(target_cloud_.points[i], pixel_limit)) {
			visible_cloud.points[size].x = target_cloud_.points[i].x;
			visible_cloud.points[size].y = target_cloud_.points[i].y;
			visible_cloud.points[size].z = target_cloud_.points[i].z;
			color = config::carla_to_cityscapes_palette_map.at(target_cloud_.points[i].label);
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

/* gets the majority vote based on the results of a knn search */
size_t SemanticCloud::GetMajorityVote(const std::vector<size_t>& knn_indices,
									  const std::vector<double>& distances) const {
	// map from semantic id to count
	std::unordered_map<unsigned int, double> map(knn_indices.size());
	std::vector<unsigned int> classes;
	double max_weight = 0;
	size_t winner_index = 0;
	for (uint32_t i = 0; i < knn_indices.size(); ++i) {
		auto point = target_cloud_.points[knn_indices[i]];
		unsigned int semantic_id = point.label;
		auto it = map.find(semantic_id);
		double additive_weight = config::semantic_weight.at(semantic_id) +
								 (1.0 / std::sqrt(distances[i])) + point.z * 100.0;
		if (it != map.end()) {
			map[semantic_id] = it->second + additive_weight;
			if (it->second + additive_weight > max_weight) {
				max_weight = it->second + additive_weight;
				winner_index = knn_indices[i];
			}
		} else {
			map[semantic_id] = additive_weight;
			if (max_weight == 0) {
				max_weight = additive_weight;
				winner_index = knn_indices[i];
			}
		}
	}
	return winner_index;
}

/* saves the color coded point cloud */
void SemanticCloud::SaveCloud(const std::string& path) const {
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
		color = config::carla_to_cityscapes_palette_map.at(target_cloud_.points[i].label);
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

/* converts the smenatic mat to an RGB image */
cv::Mat SemanticCloud::ConvertToCityScapesPallete(cv::Mat semantic_ids) {
	cv::Mat rgb_mat(semantic_ids.rows, semantic_ids.cols, CV_8UC3);
	#pragma omp parallel for
	for (int i = 0; i < semantic_ids.rows; ++i) {
		cv::Vec3b* rgb_row = rgb_mat.ptr<cv::Vec3b>(i);
		uchar* semantic_row = semantic_ids.ptr<uchar>(i);
		for (int j = 0; j < semantic_ids.cols; ++j) {
			rgb_row[j] = config::carla_to_cityscapes_palette_map.at(semantic_row[j]);
		}
	}
	return rgb_mat;
}

/* debug function that returns the x,y boundaries of the cloud */
std::tuple<double, double, double, double> SemanticCloud::GetBoundaries() const {
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

} // namespace geom