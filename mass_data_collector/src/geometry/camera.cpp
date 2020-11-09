#include "geometry/camera.h"
#include "Eigen/src/Core/Matrix.h"
#include "config/geom_config.h"

#define RAD(x) x * config::kToRadians;
namespace geom
{
Camera::Camera(boost::shared_ptr<carla::client::Sensor> sensor, float fov, // NOLINT
               unsigned int width, unsigned int height) 
        : width_(static_cast<float>(width)), height_(static_cast<float>(height)) {
    // creating transform matrix
    auto sensor_tf = sensor->GetTransform();
    Eigen::Matrix3f rot; 
    // left handed -> y & pitch flipped
    rot =  Eigen::AngleAxisf(sensor_tf.rotation.roll    * config::kToRadians * M_PI, Eigen::Vector3f::UnitX()) *
           Eigen::AngleAxisf(-sensor_tf.rotation.pitch  * config::kToRadians * M_PI, Eigen::Vector3f::UnitY()) *
           Eigen::AngleAxisf(sensor_tf.rotation.yaw     * config::kToRadians * M_PI, Eigen::Vector3f::UnitZ());
    Eigen::Vector3f trans(sensor_tf.location.x, -sensor_tf.location.y, sensor_tf.location.z);
    transform_.setIdentity();
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
    inv_transform_ = transform_.inverse();
    // creating calibration matrix :
    float f = static_cast<float>(width) / (2 * std::tan(fov * config::kToRadians / 2));
    float cx = static_cast<float>(width) / 2.0f; // NOLINT
    float cy = static_cast<float>(height) / 2.0f; // NOLINT
    kalibration_ << f, 0, cx,
                    0, f, cy,
                    0, 0, 1;
    // direct linear transform matrix;
    dlt_matrix_.setZero();
    dlt_matrix_.block<3, 3>(0, 0) = rot.transpose();
    dlt_matrix_.block<3, 1>(0, 3) = -rot.transpose() * trans;
    dlt_matrix_ = kalibration_ * dlt_matrix_;
}
/* returns the transform matrix, i.e. camera w.r.t. world */
Eigen::Matrix4f Camera::GetTransform() const {
    return transform_;
}
/* returns the inverse transform matrix, i.e. world w.r.t. camera */
Eigen::Matrix4f Camera::GetInvTransform() const {
    return inv_transform_;
}
/* returns the calibration matrix */
Eigen::Matrix3f Camera::GetCalibrationMatrix() const {
    return kalibration_;
}
/* returns whether the given point is in the view or not, given the pixel threshold */
bool Camera::IsInView(const Eigen::Vector3f& point3d, float pixel_threshold) const {
    auto res = (dlt_matrix_ * point3d.homogeneous()).normalized();
    bool in_width = res.x() >= -pixel_threshold && res.x() <= width_ + pixel_threshold;
    bool in_height = res.y() >= -pixel_threshold && res.y() <= height_ + pixel_threshold;
    return in_width && in_height;
}
/* retursn the result of a direct linear transform */
Eigen::Vector2f Camera::DLT(const Eigen::Vector3f& point3d) const {
    auto res = (dlt_matrix_ * point3d.homogeneous()).normalized();
    return Eigen::Vector2f(res.x(), res.y());
}

} // namespace geom