#ifndef __MASS_CAMERA_H__
#define __MASS_CAMERA_H__

#include <Eigen/Dense>
#include <carla/client/Sensor.h>
#include <pcl/point_types.h>

namespace geom 
{

class CameraGeometry
{
public:
    CameraGeometry(boost::shared_ptr<carla::client::Sensor> sensor, double fov, unsigned int width, unsigned int height);
    [[nodiscard]] Eigen::Matrix4d GetTransform() const;
    [[nodiscard]] Eigen::Matrix4d GetInvTransform() const;
    [[nodiscard]] Eigen::Matrix3d kalib() const;
    [[nodiscard]] Eigen::Matrix3d kalib_inv() const;
    [[nodiscard]] double width() const { return width_; }
    [[nodiscard]] double height() const { return height_; }
    
    [[nodiscard]] Eigen::Vector2d DLT(const Eigen::Vector3d& point3d) const;
    [[nodiscard]] Eigen::Vector3d Reproject(const Eigen::Vector2d& pixel_coords, double depth) const;
    [[nodiscard]] bool IsInView(const pcl::PointXYZRGB& point3d, double pixel_threshold) const;
private:
    Eigen::Matrix3d kalibration_;
    Eigen::Matrix3d inv_kalibration_;
    Eigen::Matrix<double, 3, 4> dlt_matrix_;
    Eigen::Matrix4d transform_, inv_transform_;
    double width_, height_;
};

} // namespace geom
#endif