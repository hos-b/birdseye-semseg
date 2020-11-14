#include "geometry/semantic_cloud.h"
#include "boost/algorithm/string/erase.hpp"
#include "config/geom_config.h"

#include <functional>
#include <opencv2/core/eigen.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <pcl/io/pcd_io.h>

#define RGB_MAX 255.0
namespace geom
{
SemanticCloud::SemanticCloud() {
    std::cout << "hello" << std::endl;
}
/* converts all pixels in semantic image and their depth into 3D points and adds them to the point cloud  */
void SemanticCloud::AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry, // NOLINT
                                          const Eigen::Matrix4d& car_transform,
                                          cv::Mat semantic,
                                          cv::Mat depth) {
    Eigen::Vector3d pixel_3d_loc;
    cv::Vec3b pixel_color_uc;
    pcl::PointXYZRGB point;
    size_t old_size = cloud_.points.size();
    size_t index = old_size;
    cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
    for(int i = 0; i < semantic.rows; ++i) {
        for(int j = 0; j < semantic.cols; ++j) {
            pixel_color_uc = semantic.at<cv::Vec3b>(i, j);
            cloud_[index].r = pixel_color_uc[0]; // NOLINT
            cloud_[index].g = pixel_color_uc[1]; // NOLINT
            cloud_[index].b = pixel_color_uc[2]; // NOLINT
            pixel_3d_loc = geometry->Reproject(Eigen::Vector2d(i, j), car_transform, depth.at<float>(i, j));
            cloud_[index].x = pixel_3d_loc.x(); // NOLINT
            cloud_[index].y = pixel_3d_loc.y(); // NOLINT
            cloud_[index].z = pixel_3d_loc.z(); // NOLINT
            ++index;   
        }
    }
    cloud_.width = cloud_.points.size();
    cloud_.height = 1;
    cloud_.is_dense = true;
}
/* removes all the points in the point cloud that are not visible in the specified camera geometry */
pcl::PointCloud<pcl::PointXYZRGB> SemanticCloud::MaskOutlierPoints(std::shared_ptr<geom::CameraGeometry> geometry, // NOLINT
                                      const Eigen::Matrix4d& car_transform) {
    pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;
    for (auto& point : cloud_.points) {
        if (geometry->IsInView(point, car_transform, config::kPixelDistanceThreshold)) {
            filtered_cloud.points.emplace_back(point);
        }    
    }
    filtered_cloud.width = filtered_cloud.points.size();
    filtered_cloud.height = 1;
    filtered_cloud.is_dense = true;
    std::cout << filtered_cloud.points.size() << "/" << cloud_.points.size() << " survived" << std::endl;
    // TODO(hosein): move semantics doesn't work! we'll switch to masks anyawy but wtf!
    return filtered_cloud;
}
/* returns the orthographic bird's eye view image */
cv::Mat SemanticCloud::GetBEV() const { // NOLINT
    return cv::Mat();
}

void SemanticCloud::SaveCloud(const std::string& path) {
    if (cloud_.empty()) {
        std::cout << "empty cloud, not saving." << std::endl;
        return;
    }
    pcl::io::savePCDFile(path, cloud_);
}

} // namespace geom