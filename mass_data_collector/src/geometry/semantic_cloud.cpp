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
void SemanticCloud::AddSemanticDepthImage(std::shared_ptr<geom::CameraGeometry> geometry, cv::Mat semantic, cv::Mat depth) { // NOLINT
    Eigen::Vector3d pixel_3d_loc;
    cv::Vec3b pixel_color_uc;
    pcl::PointXYZRGB point;
    size_t old_size = cloud_.points.size();
    size_t index = old_size;
    cloud_.points.resize(old_size + (semantic.rows * semantic.cols));
    for(int i = 0; i < semantic.rows; ++i) {
        for(int j = 0; j < semantic.cols; ++j) {
            pixel_color_uc = semantic.at<cv::Vec3b>(i, j);
            // cloud_.points[index++]
            cloud_[index].r = pixel_color_uc[0]; // NOLINT
            cloud_[index].g = pixel_color_uc[1]; // NOLINT
            cloud_[index].b = pixel_color_uc[2]; // NOLINT
            pixel_3d_loc = geometry->Reproject(Eigen::Vector2d(i, j), depth.at<float>(i, j));
            cloud_[index].x = pixel_3d_loc.x(); // NOLINT
            cloud_[index].y = pixel_3d_loc.y(); // NOLINT
            cloud_[index].z = pixel_3d_loc.z(); // NOLINT
            ++index;   
        }
    }
}
/* removes all the points in the point cloud that are not visible in the specified camera geometry */
void SemanticCloud::MaskOutlierPoints(std::shared_ptr<geom::CameraGeometry> geometry) { // NOLINT
    auto it = cloud_.points.begin();
    size_t erased = 0;
    size_t survived = 0;
    while (it != cloud_.points.end()) {
        if (geometry->IsInView(*it, config::kPixelDistanceThreshold)) {
            ++it;
            ++survived;
        } else {
            it = cloud_.points.erase(it);
            ++erased;
        }
    }
    std::cout << erased << " points filtered out, " << survived << " survived" << std::endl;
    // cloud_.points.erase(std::remove_if(cloud_.points.begin(), 
    //                                    cloud_.points.end(),
    //                                     [&geometry](pcl::PointXYZRGB& pt) {
    //                                         bool in_view = geometry->IsInView(pt, config::kPixelDistanceThreshold);
    //                                         return !in_view;
    //                                     }),
    //                     cloud_.points.end());
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
    cloud_.width = cloud_.points.size();
    cloud_.height = 1;
    cloud_.is_dense = true;
    pcl::io::savePCDFile(path, cloud_);
}

} // namespace geom