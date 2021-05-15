#ifndef __CARLA_SENSOR_STRUCTS__
#define __CARLA_SENSOR_STRUCTS__


#include "geometry/camera_geomtry.h"
#include "config/agent_config.h"
//------------------------------------
#include <Eigen/src/Core/Matrix.h>
#include <mutex>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//------------------------------------
#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/Image.h>
//------------------------------------
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>


namespace csd = carla::sensor::data;
namespace data
{

// conversion prototypes
cv::Mat DecodeToDepthMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image);
cv::Mat DecodeToLogarithmicDepthMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image);
cv::Mat DecodeToSemSegMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image);
cv::Mat DecodeToCityScapesPalleteSemSegMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image);

/* camera struct for carla along with required vars */
class RGBCamera {
public:
	RGBCamera(const YAML::Node& rgb_cam_node,
			  boost::shared_ptr<class carla::client::BlueprintLibrary> bp_library,
			  boost::shared_ptr<carla::client::Vehicle> vehicle,				
			  float x_cam_shift, bool log = true);
	void CaputreOnce();
	void Destroy();
	void PauseCallback();
	void ResumeCallback();
	bool waiting() const;
	size_t count() const;
	std::shared_ptr<geom::CameraGeometry> geometry() const;
	std::pair <bool, cv::Mat> pop();
private:
	std::atomic<bool> save_;
	boost::shared_ptr<carla::client::Sensor> sensor_;
	std::vector<cv::Mat> images_;
	std::shared_ptr<geom::CameraGeometry> geometry_;
	std::function<void(const boost::shared_ptr<carla::sensor::SensorData>& data)> rgb_callback_;
};

class SemanticPointCloudCamera {
public:
	SemanticPointCloudCamera(const YAML::Node& mass_cam_node,
			boost::shared_ptr<class carla::client::BlueprintLibrary> bp_library,
			boost::shared_ptr<carla::client::Vehicle> vehicle,					
			float delta_x,
			float delta_y,
			bool log = true);
	void PauseCallback();
	void ResumeCallback();
	void CaputreOnce();
	void Destroy();

	std::string name() const;
	size_t count() const;
	bool waiting() const;
	size_t depth_image_count() const;
	size_t semantic_image_count() const;
	std::shared_ptr<geom::CameraGeometry> geometry() const;
	std::tuple<bool, cv::Mat, cv::Mat> pop();
private:
	std::string name_;
	std::atomic<bool> save_depth_;
	std::atomic<bool> save_semantics_;
	boost::shared_ptr<carla::client::Sensor> depth_sensor_;
	boost::shared_ptr<carla::client::Sensor> semantic_sensor_;
	std::function<void(const boost::shared_ptr<carla::sensor::SensorData>& data)> depth_callback_;
	std::function<void(const boost::shared_ptr<carla::sensor::SensorData>& data)> semantic_callback_;
	std::shared_ptr<geom::CameraGeometry> geometry_;
	std::vector<cv::Mat> semantic_images_;
	std::vector<cv::Mat> depth_images_;
};



} // namespace data
#endif