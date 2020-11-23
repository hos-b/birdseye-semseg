#ifndef __MASS_CAMERA_H__
#define __MASS_CAMERA_H__

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <carla/client/Sensor.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
namespace geom 
{

class CameraGeometry
{
public:
    explicit CameraGeometry(const YAML::Node& cam_node, float x, float y, float z, float rr, float pp, float yy);
    [[nodiscard]] Eigen::Vector2d Test(double x, double y, double depth, const Eigen::Matrix4d& car_transform) const;
    [[nodiscard]] Eigen::Matrix4d GetTransform() const;
    [[nodiscard]] Eigen::Matrix4d GetInvTransform() const;
    [[nodiscard]] Eigen::Matrix3d kalib() const;
    [[nodiscard]] Eigen::Matrix3d kalib_inv() const;
    [[nodiscard]] double width() const { return width_; }
    [[nodiscard]] double height() const { return height_; }
    
    [[nodiscard]] Eigen::Vector2d DLT(const Eigen::Vector3d& point3d, const Eigen::Matrix4d& car_transform) const;
    [[nodiscard]] Eigen::Vector3d ReprojectToGlobal(const Eigen::Vector2d& pixel_coords, const Eigen::Matrix4d& car_transform, double depth) const;
    [[nodiscard]] Eigen::Vector3d ReprojectToLocal(const Eigen::Vector2d& pixel_coords, double depth) const;
    [[nodiscard]] bool IsInView(const pcl::PointXYZRGB& point3d, const Eigen::Matrix4d& car_transform, double pixel_threshold) const;
    [[nodiscard]] bool IsInView(const pcl::PointXYZRGB& point3d, double pixel_threshold) const;
private:
    Eigen::Matrix3d kalibration_;
    Eigen::Matrix3d inv_kalibration_;
    Eigen::Matrix<double, 3, 4> dlt_matrix_;
    Eigen::Matrix4d cam_transform_, cam_inv_transform_;
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d translation_;
    double width_, height_;
};

} // namespace geom
#endif