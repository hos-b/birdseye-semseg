#ifndef __CARLA_SENSOR_STRUCTS__
#define __CARLA_SENSOR_STRUCTS__

#include <carla/sensor/data/Image.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <carla/pointcloud/PointCloudIO.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/client/Sensor.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

/* camera struct for carla along with required vars */
struct CarlaCamera {
	sensor_msgs::CameraInfo cam_info;
	std_msgs::Header header;
    image_transport::Publisher rgb_publisher;
    ros::Publisher info_publisher;
    cv::Mat carla_image;
	boost::shared_ptr<carla::client::Sensor> sensor;
    float margin_x;
    float shift_x;
    float margin_y;
    float shift_y;
};

struct CarlaDepthCamera : public CarlaCamera {
    image_transport::Publisher float_publisher;
    boost::shared_ptr<carla::client::Sensor> sensor;
    Eigen::Matrix<float, 3, 3> intrinsics;
};

struct CarlaSemanticCamera : public CarlaCamera {
    image_transport::Publisher raw_publisher;
    cv::Mat carla_image;
    boost::shared_ptr<carla::client::Sensor> sensor;
    ros::Time prev_time_stamp;
    Eigen::Matrix<float, 3, 3> intrinsics;
};

typedef pcl::PointCloud<pcl::PointXYZ> CarlaPointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> CarlaRGBPointCloud;
struct CarlaLidar {
    boost::shared_ptr<carla::client::Sensor> sensor;
    CarlaPointCloud::Ptr data_ptr;
	ros::Publisher publisher;
    CarlaRGBPointCloud pcl_message;
};

#endif