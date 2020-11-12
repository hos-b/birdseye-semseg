#include "geometry/camera_geomtry.h"
#include "config/geom_config.h"

#define RAD(x) x * config::kToRadians;
namespace geom
{
CameraGeometry::CameraGeometry(boost::shared_ptr<carla::client::Sensor> sensor, double fov, // NOLINT
               unsigned int width, unsigned int height) 
        : width_(static_cast<double>(width)), height_(static_cast<double>(height)) {
    // creating transform matrix
    auto sensor_tf = sensor->GetTransform();
    Eigen::Matrix3d rot; 
    // left handed -> y & pitch flipped
    rot =  Eigen::AngleAxisd(sensor_tf.rotation.roll   * config::kToRadians * M_PI, Eigen::Vector3d::UnitX()) *
           Eigen::AngleAxisd(-sensor_tf.rotation.pitch * config::kToRadians * M_PI, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(sensor_tf.rotation.yaw    * config::kToRadians * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(sensor_tf.location.x, -sensor_tf.location.y, sensor_tf.location.z);
    transform_.setIdentity();
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
    inv_transform_ = transform_.inverse();
    // creating calibration matrix :
    double f = static_cast<double>(width) / (2 * std::tan(fov * config::kToRadians / 2));
    double cx = static_cast<double>(width) / 2.0f; // NOLINT
    double cy = static_cast<double>(height) / 2.0f; // NOLINT
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
    return transform_;
}
/* returns the inverse transform matrix, i.e. transforms camera to world */
Eigen::Matrix4d CameraGeometry::GetInvTransform() const {
    return inv_transform_;
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
bool CameraGeometry::IsInView(const pcl::PointXYZRGB& point3d, double pixel_threshold) const {
    auto local = (transform_ * Eigen::Vector4d(point3d.x, point3d.y, point3d.z, 1.0)).normalized(); // NOLINT
    auto res = (kalibration_ * local.head<3>()); // NOLINT
    // auto res = (dlt_matrix_ * Eigen::Vector4d(point3d.x, point3d.y, point3d.z, 1.0)).normalized(); // NOLINT
    bool in_width = res.x() >= -pixel_threshold && res.x() <= width_ + pixel_threshold;
    bool in_height = res.y() >= -pixel_threshold && res.y() <= height_ + pixel_threshold;
    return in_width && in_height;
}
/* returns the result of a direct linear transform */
Eigen::Vector2d CameraGeometry::DLT(const Eigen::Vector3d& point3d) const {
    auto res = (dlt_matrix_ * point3d.homogeneous()).normalized();
    return Eigen::Vector2d(res.x(), res.y());
}
/* reprojects the point back into the global 3D coordinates */
Eigen::Vector3d CameraGeometry::Reproject(const Eigen::Vector2d& pixel_coords, double depth) const {
    auto local = inv_kalibration_ * pixel_coords.homogeneous() * depth;
    auto global = inv_transform_ * local.homogeneous();
    return global.head<3>() / global.w();
}

} // namespace geom