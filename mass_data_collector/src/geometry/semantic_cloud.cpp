#include "geometry/semantic_cloud.h"
#include "boost/algorithm/string/erase.hpp"
#include "config/agent_config.h"
#include "config/geom_config.h"

#include <Eigen/src/Core/Matrix.h>
#include <functional>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <tuple>
#include <unordered_map>
#include <pcl/io/pcd_io.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/viz/vizcore.hpp>

#define RGB_MAX 255.0
namespace geom
{
SemanticCloud::SemanticCloud(double max_x, double max_y, size_t img_rows, size_t img_cols) {
    point_max_x_ = max_x;
    point_max_y_ = max_y;
    image_rows_ = img_rows;
    image_cols_ = img_cols;
    kd_tree_ = nullptr;
    xy_hash_ = [](const std::pair<double, double>& pair) -> size_t {
        return std::hash<double>()(pair.first * 1e4) ^ std::hash<double>()(pair.second); // NOLINT
    };
    semantic_color_hash_ = [](const Eigen::Vector3i& rgb) -> size_t {
        return std::hash<int>()((rgb.x() << 16 ) | (rgb.y() << 8) | rgb.z());
    };
}
/* converts all pixels in semantic image and their depth into 3D points [in the car transform] and adds them to the point cloud  */
void SemanticCloud::AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry, // NOLINT
                                          cv::Mat semantic,
                                          cv::Mat depth) {
    Eigen::Vector3d pixel_3d_loc;
    cv::Vec3b pixel_color_uc;
    pcl::PointXYZRGB point;
    size_t old_size = target_cloud_.points.size();
    size_t index = old_size;
    target_cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
    for (int i = 0; i < semantic.rows; ++i) {
        for (int j = 0; j < semantic.cols; ++j) {
            pixel_color_uc = semantic.at<cv::Vec3b>(i, j);
            target_cloud_[index].r = pixel_color_uc[0]; // NOLINT
            target_cloud_[index].g = pixel_color_uc[1]; // NOLINT
            target_cloud_[index].b = pixel_color_uc[2]; // NOLINT
            pixel_3d_loc = geometry->ReprojectToLocal(Eigen::Vector2d(i, j), depth.at<float>(i, j));
            target_cloud_[index].x = pixel_3d_loc.x(); // NOLINT
            target_cloud_[index].y = pixel_3d_loc.y(); // NOLINT
            target_cloud_[index].z = pixel_3d_loc.z(); // NOLINT
            ++index;   
        }
    }
    target_cloud_.width = target_cloud_.points.size();
    target_cloud_.height = 1;
    target_cloud_.is_dense = true;
}
/* removes overlapping or invisible points for the target cloud. additionally removes points not
   visible in the rgb camera for the masked cloud
*/
void SemanticCloud::FilterCloud(/* std::shared_ptr<geom::CameraGeometry> rgb_geometry, // NOLINT
                                double pixel_limit */) {
    std::unordered_map<std::pair<double, double>, double, decltype(xy_hash_)> map(10000, xy_hash_);
    pcl::PointCloud<pcl::PointXYZRGB> new_target_cloud;
    for (auto& org_point : target_cloud_.points) {
        Eigen::Vector4d local_car = Eigen::Vector4d(org_point.x, org_point.y, org_point.z, 1.0); // NOLINT
        // skip if outside the given boundaries
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
    std::cout << "filtered " << target_cloud_.points.size() - new_target_cloud.points.size()
              << "/" << target_cloud_.points.size() << " points" << std::endl;
    target_cloud_ = new_target_cloud;
}
/* returns the orthographic bird's eye view image */
std::pair <cv::Mat, cv::Mat> SemanticCloud::GetSemanticBEV(std::shared_ptr<geom::CameraGeometry> rgb_geometry,
        double pixel_limit, double mask_dist_threshold) {
    double pixel_w = 2 * point_max_y_ / static_cast<double>(image_cols_);
    double pixel_h = 2 * point_max_x_ / static_cast<double>(image_rows_);
    // cam stuff
    auto cam_inv_transform = rgb_geometry->GetInvTransform();
    auto cam_kalib = rgb_geometry->kalib();
    auto cam_img_height = rgb_geometry->height();
    auto cam_img_width = rgb_geometry->width();
    cv::Mat semantic_bev(image_rows_, image_cols_, CV_8UC3);
    cv::Mat semantic_mask = cv::Mat::zeros(image_rows_, image_cols_, CV_8UC1);
    // building kd-tree
    kd_tree_ = new nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, SemanticCloud>,
                    SemanticCloud, 2> (2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    std::cout << "kd-tree build" << std::endl;
    kd_tree_->buildIndex();
    for (int i = 0; i < semantic_bev.rows; ++i) {
        for (int j = 0; j < semantic_bev.cols; ++j) {
            // get the center of a square in the point cloud that is to be mapped to a pixel
            double knn_x = point_max_x_ - i * pixel_h + 0.5 * pixel_h;
            double knn_y = point_max_y_ - j * pixel_w + 0.5 * pixel_w;
            double query[2] = {knn_x, knn_y};
            size_t num_results = 5;
            std::vector<size_t> knn_ret_index(num_results);
            std::vector<double> knn_sqrd_dist(num_results);
            num_results = kd_tree_->knnSearch(&query[0], num_results, &knn_ret_index[0], &knn_sqrd_dist[0]);
            // in case of less points in the tree than requested:
            knn_ret_index.resize(num_results);
            knn_sqrd_dist.resize(num_results);
            // majority voting
            std::unordered_map<const Eigen::Vector3i, size_t, decltype(semantic_color_hash_)> map(num_results, semantic_color_hash_);
            size_t max = 0;
            size_t max_index = 0;
            for (size_t index : knn_ret_index) {
                auto point = target_cloud_.points.at(index);
                auto rgbvec = Eigen::Vector3i(point.r, point.g, point.b);
                auto it = map.find(rgbvec);
                if (it != map.end()) {
                    map[Eigen::Vector3i(point.r, point.g, point.b)] = it->second + 1;
                    if (it->second + 1 > max) {
                        max = it->second + 1;
                        max_index = index;
                    }
                } else {
                    map[rgbvec] = 1;
                    if (max == 0) {
                        max = 1;
                        max_index = index;
                    }
                }
            }
            auto mv_winner = target_cloud_.points[max_index];
            semantic_bev.at<cv::Vec3b>(i, j) = cv::Vec3b(mv_winner.r, mv_winner.g, mv_winner.b);
            // if closest point is in threshold, check for visibilty
            if (std::sqrt(knn_sqrd_dist[0]) <= mask_dist_threshold) {
                auto point = target_cloud_.points[knn_ret_index[0]];
                Eigen::Vector4d local_car = Eigen::Vector4d(point.x, point.y, point.z, 1.0);
                Eigen::Vector3d local_camera = (cam_inv_transform * local_car).hnormalized();
                // TODO(me): add the car iteself
                // points behind the camera don't count
                if (local_camera.z() < 0) {
                    continue;
                }
                Eigen::Vector2d prj = (cam_kalib * local_camera).hnormalized();
                bool in_height = prj.x() >= -pixel_limit && prj.x() <= cam_img_height + pixel_limit;
                bool in_width = prj.y() >= -pixel_limit && prj.y() <= cam_img_width + pixel_limit;
                // if outside the rgb image boundaries, skip
                if (!in_width || !in_height) {
                    continue;
                }
                semantic_mask.at<unsigned char>(i, j) = 255;
            }
        }
    }
    std::cout << "deleting kd-tree" << std::endl;
    delete kd_tree_;
    kd_tree_ = nullptr;
    return std::make_pair(semantic_bev, semantic_mask);
}
/* returns the points that are visible given the camera geometry */
/* pcl::PointCloud<pcl::PointXYZRGB> SemanticCloud::GetMaskedCloud(std::shared_ptr<geom::CameraGeometry> rgb_geometry) {
    pcl::PointCloud<pcl::PointXYZRGB> masked;

    return masked;
} */
/* saves the target point cloud after checking if it's empty */
void SemanticCloud::SaveTargetCloud(const std::string& path) {
    if (target_cloud_.empty()) {
        std::cout << "empty cloud, not saving." << std::endl;
        return;
    }
    pcl::io::savePCDFile(path, target_cloud_);
}
} // namespace geom