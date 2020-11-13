#include "geometry/camera_geomtry.h"
#include "config/geom_config.h"

#define RAD(x) x * config::kToRadians;
namespace geom
{
CameraGeometry::CameraGeometry(const YAML::Node& cam_node) {
    width_ = cam_node["image_size_x"].as<double>();
    height_ = cam_node["image_size_y"].as<double>();
    // creating transform matrix
    Eigen::Matrix3d rot; 
    // left handed -> y & pitch flipped
    rot =  Eigen::AngleAxisd(cam_node["roll"].as<double>()   * config::kToRadians * M_PI, Eigen::Vector3d::UnitX()) *
           Eigen::AngleAxisd(-cam_node["pitch"].as<double>() * config::kToRadians * M_PI, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(cam_node["yaw"].as<double>()    * config::kToRadians * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(cam_node["x"].as<double>(), -cam_node["y"].as<double>(), cam_node["z"].as<double>());
    cam_transform_.setIdentity();
    cam_transform_.block<3, 3>(0, 0) = rot;
    cam_transform_.block<3, 1>(0, 3) = trans;
    cam_inv_transform_ = cam_transform_.inverse();
    // creating calibration matrix :
    double f = width_ / (2 * std::tan(cam_node["fov"].as<double>() * config::kToRadians / 2));
    double cx = width_ / 2.0f; // NOLINT
    double cy = height_ / 2.0f; // NOLINT
    kalibration_ << f, 0, cx,
                    0, f, cy,
                    0, 0, 1;
    inv_kalibration_ = kalibration_.inverse();
    // direct linear transform matrix;
    dlt_matrix_.setZero();
    dlt_matrix_.block<3, 3>(0, 0) = rot.transpose();
    dlt_matrix_.block<3, 1>(0, 3) = -rot.transpose() * trans;
    dlt_matrix_ = kalibration_ * dlt_matrix_;
}
/* returns the transform matrix, i.e. transforms world to camera */
Eigen::Matrix4d CameraGeometry::GetTransform() const {
    return cam_transform_;
}
/* returns the inverse transform matrix, i.e. transforms camera to world */
Eigen::Matrix4d CameraGeometry::GetInvTransform() const {
    return cam_inv_transform_;
}
/* returns the calibration matrix */
Eigen::Matrix3d CameraGeometry::kalib() const {
    return kalibration_;
}
/* returns the inverse calibration matrix */
Eigen::Matrix3d CameraGeometry::kalib_inv() const {
    return inv_kalibration_;
}
/* returns whether the given point is in the view or not, given the pixel threshold */
bool CameraGeometry::IsInView(const pcl::PointXYZRGB& point3d, const Eigen::Matrix4d& car_transform, double pixel_threshold) const {
    auto local_car = car_transform.inverse() * Eigen::Vector4d(point3d.x, point3d.y, point3d.z, 1.0); // NOLINT
    auto local_cam = (cam_inv_transform_ * local_car).normalized(); 
    auto res = (kalibration_ * local_cam.head<3>()); // NOLINT
    bool in_width = res.x() >= -pixel_threshold && res.x() <= width_ + pixel_threshold;
    bool in_height = res.y() >= -pixel_threshold && res.y() <= height_ + pixel_threshold;
    return in_width && in_height;
}
/* returns the result of a direct linear transform */
Eigen::Vector2d CameraGeometry::DLT(const Eigen::Vector3d& point3d, const Eigen::Matrix4d& car_transform) const {
    auto car_local = car_transform.inverse() * point3d.homogeneous(); // NOLINT
    auto res = (dlt_matrix_ * car_local).normalized();
    return Eigen::Vector2d(res.x(), res.y());
}
/* reprojects the point back into the global 3D coordinates */
Eigen::Vector3d CameraGeometry::Reproject(const Eigen::Vector2d& pixel_coords, const Eigen::Matrix4d& car_transform, double depth) const {
    auto cam_local = inv_kalibration_ * pixel_coords.homogeneous() * depth;
    auto global = car_transform * cam_transform_ * cam_local.homogeneous();
    return global.head<3>() / global.w();
}

} // namespace geom