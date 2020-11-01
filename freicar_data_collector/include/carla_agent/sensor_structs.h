#ifndef __CARLA_SENSOR_STRUCTS__
#define __CARLA_SENSOR_STRUCTS__

#include <carla/sensor/data/Image.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/client/Sensor.h>


/* camera struct for carla along with required vars */
struct CarlaCamera {
    cv::Mat carla_image;
	boost::shared_ptr<carla::client::Sensor> sensor;
    float margin_x;
    float shift_x;
    float margin_y;
    float shift_y;
};

struct CarlaDepthCamera : public CarlaCamera {
    boost::shared_ptr<carla::client::Sensor> sensor;
    Eigen::Matrix<float, 3, 3> intrinsics;
};

struct CarlaSemanticCamera : public CarlaCamera {
    cv::Mat carla_image;
    boost::shared_ptr<carla::client::Sensor> sensor;
    Eigen::Matrix<float, 3, 3> intrinsics;
};


#endif