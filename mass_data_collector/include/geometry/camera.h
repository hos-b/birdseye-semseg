#ifndef __MASS_CAMERA_H__
#define __MASS_CAMERA_H__

#include <Eigen/Dense>
#include <carla/client/Sensor.h>

namespace geom 
{

class Camera
{
public:
    Camera(boost::shared_ptr<carla::client::Sensor> sensor, float fov, unsigned int width, unsigned int height);
    [[nodiscard]] Eigen::Matrix4f GetTransform() const;
    [[nodiscard]] Eigen::Matrix4f GetInvTransform() const;
    [[nodiscard]] Eigen::Matrix3f GetCalibrationMatrix() const;
    [[nodiscard]] float width() const { return width_; }
    [[nodiscard]] float height() const { return height_; }
    
    [[nodiscard]] Eigen::Vector2f DLT(const Eigen::Vector3f& point3d) const;
    [[nodiscard]] bool IsInView(const Eigen::Vector3f& point3d, float pixel_threshold) const;
private:
    Eigen::Matrix3f kalibration_;
    Eigen::Matrix<float, 3, 4> dlt_matrix_;
    Eigen::Matrix4f transform_, inv_transform_;
    float width_, height_;
};

} // namespace geom
#endif