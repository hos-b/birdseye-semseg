#include "geometry/semantic_cloud.h"
#include "boost/algorithm/string/erase.hpp"
#include "config/agent_config.h"
#include "config/geom_config.h"

#include <cmath>
#include <functional>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>
#include <unordered_map>
#include <pcl/io/pcd_io.h>

#define RGB_MAX 255.0
namespace geom
{
/* constructor */
SemanticCloud::SemanticCloud(double max_x, double max_y, size_t img_rows, size_t img_cols) {
	point_max_x_ = max_x;
	point_max_y_ = max_y;
	image_rows_ = img_rows;
	image_cols_ = img_cols;
	kd_tree_ = nullptr;
	xy_hash_ = [](const std::pair<double, double>& pair) -> size_t {
		return std::hash<double>()(pair.first * 1e4) ^ std::hash<double>()(pair.second);
	};
	pixel_w_ = 2 * point_max_y_ / static_cast<double>(image_cols_);
	pixel_h_ = 2 * point_max_x_ / static_cast<double>(image_rows_);
}
/* destructor: cleans containers */
SemanticCloud::~SemanticCloud() {
	if (kd_tree_) {
		kd_tree_.reset();
		kd_tree_ = nullptr;
	}
	target_cloud_.points.clear();
}
/* converts all pixels in semantic image and their depth into 3D points [in the car transform] and adds them to the point cloud  */
void SemanticCloud::AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry,
										  cv::Mat semantic,
										  cv::Mat depth) {
	Eigen::Vector3d pixel_3d_loc;
	size_t old_size = target_cloud_.points.size();
	size_t index = old_size;
	target_cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
	for (int i = 0; i < semantic.rows; ++i) {
		for (int j = 0; j < semantic.cols; ++j) {
			target_cloud_[index].label = semantic.at<unsigned char>(i, j);
			pixel_3d_loc = geometry->ReprojectToLocal(Eigen::Vector2d(i, j), depth.at<float>(i, j));
			target_cloud_[index].x = pixel_3d_loc.x();
			target_cloud_[index].y = pixel_3d_loc.y();
			target_cloud_[index].z = pixel_3d_loc.z();
			++index;
		}
	}
	target_cloud_.width = target_cloud_.points.size();
	target_cloud_.height = 1;
	target_cloud_.is_dense = true;
}
/* removes overlapping or invisible points for the target cloud. initializes kd tree.
   filters unwanted labels (declared in config::filtered_semantics)
*/
void SemanticCloud::ProcessCloud() {
	std::unordered_map<std::pair<double, double>, double, decltype(xy_hash_)> map(target_cloud_.points.size(), xy_hash_);
	pcl::PointCloud<pcl::PointXYZL> new_target_cloud;
	for (auto& org_point : target_cloud_.points) {
		// filter if it belongs to a traffic light or pole
		if (config::fileterd_semantics.at(org_point.label)) {
			continue;
		}
		Eigen::Vector4d local_car = Eigen::Vector4d(org_point.x, org_point.y, org_point.z, 1.0); // NOLINT
		// skip if outside the given BEV boundaries
		if (local_car.y() < -point_max_y_ || local_car.y() > point_max_y_ ||
			local_car.x() < -point_max_x_ || local_car.x() > point_max_x_) {
			continue;
		}
		// skip if repetitive or lower height, surprisingly this never triggers
		auto xy_pair = std::make_pair(local_car.x(), local_car.y());
		auto it = map.find(xy_pair);
		if (it != map.end() && it->second > local_car.z()) {
			continue;
		}
		map[xy_pair] = local_car.z();
		new_target_cloud.points.emplace_back(org_point);
	}
	new_target_cloud.width = new_target_cloud.points.size();
	new_target_cloud.height = 1;
	new_target_cloud.is_dense = true;
	// std::cout << "filtered " << target_cloud_.points.size() - new_target_cloud.points.size()
	// 		  << "/" << target_cloud_.points.size() << " points" << std::endl;
	target_cloud_ = new_target_cloud;
	// building kd-tree
	kd_tree_ = std::make_unique<KDTree2D>(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	kd_tree_->buildIndex();
}
/* returns the orthographic bird's eye view image */
std::pair <cv::Mat, cv::Mat> SemanticCloud::GetSemanticBEV(double vehicle_width, double vehicle_length) {
	// getting the vehicle rectangle
	cv::Point mid(image_cols_ / 2, image_rows_ / 2);
	cv::Point topleft(mid.x - (vehicle_width / (2 * pixel_w_)), mid.y - (vehicle_length / (2 * pixel_h_)));
	cv::Point botright(mid.x + (vehicle_width / (2 * pixel_w_)), mid.y + (vehicle_length / (2 * pixel_h_)));
	cv::Rect2d vhc_rect(topleft, botright);
	// the output images
	cv::Mat semantic_bev(image_rows_, image_cols_, CV_8UC1);
	cv::Mat vehicle_mask = cv::Mat::zeros(image_rows_, image_cols_, CV_8UC1);
	for (int i = 0; i < semantic_bev.rows; ++i) {
		for (int j = 0; j < semantic_bev.cols; ++j) {
			// get the center of the square that is to be mapped to a pixel
			double knn_x = point_max_x_ - i * pixel_h_ + 0.5 * pixel_h_;
			double knn_y = point_max_y_ - j * pixel_w_ + 0.5 * pixel_w_;
			// ------------------------ majority voting for semantic id ------------------------
			auto [knn_ret_index, knn_sqrd_dist] = FindClosestPoints(knn_x, knn_y, 32);
			auto mv_winner = target_cloud_.points[GetMajorityVote(knn_ret_index)];
			semantic_bev.at<unsigned char>(i, j) = static_cast<unsigned char>(mv_winner.label);
			// ------------------------ checking visibilty for the mask ------------------------
			// if the point belongs to the car itself
			if (mv_winner.label == config::kCARLAVehiclesSemanticID &&
				vhc_rect.contains(cv::Point(j, i))) {
				vehicle_mask.at<unsigned char>(i, j) = 255;
				continue;
			}
		}
	}
	return std::make_pair(semantic_bev, vehicle_mask);
}
/* returns the BEV mask visible from the camera. this method assumes the point cloud only
   consists of points captured with the front camera */
cv::Mat SemanticCloud::GetBEVMask(double stitching_threshold) {
	cv::Mat bev_mask = cv::Mat::zeros(image_rows_, image_cols_, CV_8UC1);
	for (int i = 0; i < bev_mask.rows; ++i) {
		for (int j = 0; j < bev_mask.cols; ++j) {
			// get the center of the square that is to be mapped to a pixel
			double knn_x = point_max_x_ - i * pixel_h_ + 0.5 * pixel_h_;
			double knn_y = point_max_y_ - j * pixel_w_ + 0.5 * pixel_w_;
			auto [knn_ret_index, knn_sqrd_dist] = FindClosestPoints(knn_x, knn_y, 32);
			// if the point is close enough
			if (knn_sqrd_dist[0] <= stitching_threshold) {
				bev_mask.at<unsigned char>(i, j) = 255;
			}
		}
	}
	return bev_mask;
}

/* simple kd-tree look up. only use whe the cloud is  */
std::pair<std::vector<size_t>, std::vector<double>> SemanticCloud::FindClosestPoints(double knn_x, double knn_y, size_t num_results) {
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
[[deprecated("deprecated function")]] std::tuple<double, double, double, double> SemanticCloud::GetVehicleBoundary() {
	double front = 0.0;
	double back = 0.0;
	double left = 0.0;
	double right = 0.0;
	double seek = 0.0;
	// going cm by cm along the axes to find the boundaries
	while (true) {
		seek += 0.01;
		auto [pt_indices, pt_dists] = FindClosestPoints(seek, 0, 1);
		auto& next_point = target_cloud_.points[pt_indices[0]];
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
		auto& next_point = target_cloud_.points[pt_indices[0]];
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
		auto& next_point = target_cloud_.points[pt_indices[0]];
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
		auto& next_point = target_cloud_.points[pt_indices[0]];
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
	for (auto org_point : target_cloud_.points) {
		if (rgb_geometry->IsInView(org_point, pixel_limit)) {
			visible_cloud.points[size].x = org_point.x;
			visible_cloud.points[size].y = org_point.y;
			visible_cloud.points[size].z = org_point.z;
			color = config::semantic_palette_map.at(org_point.label);
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
size_t SemanticCloud::GetMajorityVote(const std::vector<size_t>& knn_result) {
	// map from semantic id to count
	std::unordered_map<unsigned int, size_t> map(knn_result.size());
	size_t max = 0;
	size_t winner_index = 0;
	for (size_t index : knn_result) {
		auto point = target_cloud_.points.at(index);
		unsigned int semantic_id = point.label;
		auto it = map.find(semantic_id);
		if (it != map.end()) {
			map[semantic_id] = it->second + 1;
			if (it->second + 1 > max) {
				max = it->second + 1;
				winner_index = index;
			}
		} else {
			map[semantic_id] = 1;
			if (max == 0) {
				max = 1;
				winner_index = index;
			}
		}
	}
	return winner_index;
}
/* saves the color coded point cloud */
void SemanticCloud::SaveCloud(const std::string& path) {
	if (target_cloud_.empty()) {
		std::cout << "empty cloud, not saving." << std::endl;
		return;
	}
	pcl::PointCloud<pcl::PointXYZRGB> visible_cloud;
	visible_cloud.points.resize(target_cloud_.points.size());
	size_t size = 0;
	cv::Scalar color;
	for (auto org_point : target_cloud_.points) {
		visible_cloud.points[size].x = org_point.x;
		visible_cloud.points[size].y = org_point.y;
		visible_cloud.points[size].z = org_point.z;
		color = config::semantic_palette_map.at(org_point.label);
		visible_cloud.points[size].b = color[0];
		visible_cloud.points[size].g = color[1];
		visible_cloud.points[size].r = color[2];
		++size;
	}
	visible_cloud.width = size;
	visible_cloud.height = 1;
	visible_cloud.is_dense = true;
	pcl::io::savePCDFile(path, visible_cloud);
}
/* converts the smenatic mat to an RGB image */
cv::Mat SemanticCloud::ConvertToCityScapesPallete(cv::Mat semantic_ids) {
	cv::Mat rgb_mat(semantic_ids.rows, semantic_ids.cols, CV_8UC3);
	for (int i = 0; i < semantic_ids.rows; ++i) {
		for (int j = 0; j < semantic_ids.cols; ++j) {
			rgb_mat.at<cv::Vec3b>(i, j) = config::semantic_palette_map.at(semantic_ids.at<unsigned char>(i, j));
		}
	}
	return rgb_mat;
}
/* debug function that prints the x,y boundaries of the cloud */
void SemanticCloud::PrintBoundaries() {
	float minx = 0, miny = 0, maxx = 0, maxy = 0;
	for (auto& point : target_cloud_.points) {
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
	std::cout << "(" << minx << ", " << miny << ") - (" << maxx << ", " << maxy << ")" << std::endl;
}
} // namespace geom