#include "transform_conversions.h"
#include <Eigen/Dense>

#define ToRadians static_cast<float>(M_PI / 180.0f)
#define ToDegrees static_cast<float>(180.0f / M_PI)

carla::geom::Transform ApplyTransform(carla::geom::Transform tf,
                                      float delta_x,
                                      float delta_y,
                                      float delta_pitch,
                                      float delta_yaw) {
    Eigen::Matrix4f current_tf_eig;
    current_tf_eig.setIdentity();
    Eigen::Matrix3f current_rot;
    current_rot = Eigen::AngleAxisf(tf.rotation.roll  * ToRadians, Eigen::Vector3f::UnitX()) *
                  Eigen::AngleAxisf(tf.rotation.pitch * ToRadians, Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(tf.rotation.yaw   * ToRadians, Eigen::Vector3f::UnitZ());
    current_tf_eig.block<3, 3>(0, 0) = current_rot;
    current_tf_eig.block<3, 1>(0, 3) = Eigen::Vector3f(tf.location.x, tf.location.y, tf.location.z);
    current_tf_eig(3, 0) = current_tf_eig(3, 1) = 0;

    Eigen::Matrix4f delta_tf_eig;
    delta_tf_eig.setIdentity();
    Eigen::Matrix3f delta_rot;
    delta_rot = Eigen::AngleAxisf(0           * ToRadians, Eigen::Vector3f::UnitX()) *
                Eigen::AngleAxisf(delta_pitch * ToRadians, Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(delta_yaw   * ToRadians, Eigen::Vector3f::UnitZ());
    delta_tf_eig.block<3, 3>(0, 0) = delta_rot;
    delta_tf_eig.block<3, 1>(0, 3) = Eigen::Vector3f(delta_x, delta_y, 0);
    delta_tf_eig(3, 0) = delta_tf_eig(3, 1) = 0;

    auto new_eig_tf = current_tf_eig * delta_tf_eig;
    Eigen::Matrix3f rot_matix = new_eig_tf.block<3, 3>(0, 0);
    Eigen::Vector3f rpy_vec = rot_matix.eulerAngles(0, 1, 2) * ToDegrees;
    return carla::geom::Transform{carla::geom::Location{new_eig_tf(0, 3), new_eig_tf(1, 3), new_eig_tf(2, 3)},
                                  carla::geom::Rotation{rpy_vec(1), rpy_vec(2), rpy_vec(0)}};
}

Eigen::Matrix4f FromCARLA(carla::geom::Transform& tf) {
    Eigen::Matrix4f current_tf_eig;
    current_tf_eig.setIdentity();
    Eigen::Matrix3f current_rot;
    current_rot = Eigen::AngleAxisf(-tf.rotation.roll  * ToRadians, Eigen::Vector3f::UnitX()) *
                  Eigen::AngleAxisf(-tf.rotation.pitch * ToRadians, Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(-tf.rotation.yaw   * ToRadians, Eigen::Vector3f::UnitZ());
    current_tf_eig.block<3, 3>(0, 0) = current_rot;
    current_tf_eig.block<3, 1>(0, 3) = Eigen::Vector3f(tf.location.x, -tf.location.y, tf.location.z);
    current_tf_eig(3, 0) = current_tf_eig(3, 1) = 0;
    return current_tf_eig;
}

carla::geom::Transform FromEigen(Eigen::Matrix4f& eig) {
    Eigen::Matrix3f rot_matix = eig.block<3, 3>(0, 0);
    Eigen::Vector3f rpy_vec = rot_matix.eulerAngles(0, 1, 2) * ToDegrees;
    return carla::geom::Transform{carla::geom::Location{eig(0, 3), -eig(1, 3), eig(2, 3)},
                                  carla::geom::Rotation{-rpy_vec(1), -rpy_vec(2), -rpy_vec(0)}};
}