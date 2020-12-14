#include "geometry/camera_geomtry.h"
#include "config/geom_config.h"
#include <cstddef>

#define RAD(x) x * config::kToRadians;
namespace geom
{
CameraGeometry::CameraGeometry(const YAML::Node& cam_node, float x, float y, float z, float rr, float pp, float yy) {
    width_ = cam_node["image_size_x"].as<double>();
    height_ = cam_node["image_size_y"].as<double>();
    // from normal coordinates to a camera looking in +z, x down, y left
    Eigen::Matrix3d affine_rotation;
    affine_rotation = Eigen::AngleAxisd(0.0       , Eigen::Vector3d::UnitX()) *
                      Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(0.0       , Eigen::Vector3d::UnitZ());
    // creating transform matrix
    // left handed -> y & pitch flipped
    rotation_ = Eigen::AngleAxisd(rr *  config::kToRadians, Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(pp * -config::kToRadians, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(yy *  config::kToRadians, Eigen::Vector3d::UnitZ());
    rotation_ = affine_rotation * rotation_;
    translation_ = Eigen::Vector3d(x, -y, z);
    cam_transform_.setIdentity();
    cam_transform_.block<3, 3>(0, 0) = rotation_;
    cam_transform_.block<3, 1>(0, 3) = translation_;
    cam_inv_transform_ = cam_transform_.inverse();
    // creating calibration matrix :
    double f = width_ / (2 * std::tan(cam_node["fov"].as<double>() * config::kToRadians / 2));
    double cx = height_ / 2.0f;
    double cy = width_ / 2.0f;
    // had to flip y for some reason to match cyrill's coordinates
    kalibration_ << f,  0, cx,
                    0, -f, cy,
                    0,  0,  1;
    inv_kalibration_ = kalibration_.inverse();
    // direct linear transform matrix using inverse transform
    dlt_matrix_.setZero();
    dlt_matrix_.block<3, 3>(0, 0) = rotation_.transpose();
    dlt_matrix_.block<3, 1>(0, 3) = -rotation_.transpose() * translation_;
    dlt_matrix_ = kalibration_ * dlt_matrix_;
}
/* returns whether the given [global] point is in the view or not, given the pixel threshold */
bool CameraGeometry::IsInView(const pcl::PointXYZL& point3d, const Eigen::Matrix4d& car_transform, double pixel_threshold) const {
    Eigen::Vector4d local_car = car_transform.inverse() * Eigen::Vector4d(point3d.x, point3d.y, point3d.z, 1.0); // NOLINT
    Eigen::Vector3d local_camera = (cam_inv_transform_ * local_car).hnormalized();
    // points behind the camera don't count
    if (local_camera.z() < 0) {
        return false;
    }
    Eigen::Vector2d prj = (kalibration_ * local_camera).hnormalized();
    bool in_height = prj.x() >= -pixel_threshold && prj.x() <= height_ + pixel_threshold;
    bool in_width = prj.y() >= -pixel_threshold && prj.y() <= width_ + pixel_threshold;
    return in_width && in_height;
}
/* returns whether the given [local in car's transform] point is in the view or not, given the pixel threshold */
bool CameraGeometry::IsInView(const pcl::PointXYZL& point3d, double pixel_threshold) const {
    Eigen::Vector4d local_car = Eigen::Vector4d(point3d.x, point3d.y, point3d.z, 1.0);
    Eigen::Vector3d local_camera = (cam_inv_transform_ * local_car).hnormalized();
    // points behind the camera don't count
    if (local_camera.z() < 0) {
        return false;
    }
    Eigen::Vector2d prj = (kalibration_ * local_camera).hnormalized();
    bool in_height = prj.x() >= -pixel_threshold && prj.x() <= height_ + pixel_threshold;
    bool in_width = prj.y() >= -pixel_threshold && prj.y() <= width_ + pixel_threshold;
    return in_width && in_height;
}

/* returns the result of a direct linear transform */
Eigen::Vector2d CameraGeometry::DLT(const Eigen::Vector3d& point3d, const Eigen::Matrix4d& car_transform) const {
    Eigen::Vector4d local_car = car_transform.inverse() * point3d.homogeneous();
    return (dlt_matrix_ * local_car).hnormalized();
}
/* reprojects the point back into the global 3D coordinates */
Eigen::Vector3d CameraGeometry::ReprojectToGlobal(const Eigen::Vector2d& pixel_coords, const Eigen::Matrix4d& car_transform, double depth) const {
    Eigen::Vector3d cam_local = inv_kalibration_ * pixel_coords.homogeneous() * depth;
    Eigen::Vector4d global = car_transform * cam_transform_ * cam_local.homogeneous();
    return global.hnormalized();
}
/* reprojects the point back into the local car 3D coordinates */
Eigen::Vector3d CameraGeometry::ReprojectToLocal(const Eigen::Vector2d& pixel_coords, double depth) const {
    Eigen::Vector3d cam_local = inv_kalibration_ * pixel_coords.homogeneous() * depth;
    Eigen::Vector4d car_local = cam_transform_ * cam_local.homogeneous();
    return car_local.hnormalized();
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
/* testing whether points get projected to where they should */
// TODO(me): turn into unit test
Eigen::Vector2d CameraGeometry::Test(double x, double y, double depth,const Eigen::Matrix4d& car_transform) const {
    Eigen::Vector3d global_point = ReprojectToGlobal(Eigen::Vector2d(x, y), car_transform, depth);
    std::cout << "intermediate: " << global_point.transpose() << std::endl;
    return DLT(global_point, car_transform);
}

} // namespace geom