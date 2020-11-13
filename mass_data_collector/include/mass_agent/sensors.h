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
			  boost::shared_ptr<class carla::client::BlueprintLibrary> bp_library,	// NOLINT
			  boost::shared_ptr<carla::client::Vehicle> vehicle,					// NOLINT
			  bool log = true);
	void CaputreOnce();
	void Destroy();

	[[nodiscard]] bool waiting() const;
	[[nodiscard]] size_t count() const;
	[[nodiscard]] std::shared_ptr<geom::CameraGeometry> geometry() const;
	[[nodiscard]] std::pair <bool, cv::Mat> pop();
private:
	bool save_ = false;
	boost::shared_ptr<carla::client::Sensor> sensor_;
	std::vector<cv::Mat> images_;
	std::shared_ptr<geom::CameraGeometry> geometry_;
	std::mutex buffer_mutex_;
};

class SemanticPointCloudCamera {
public:
	SemanticPointCloudCamera(const YAML::Node& depth_cam_node,
			const YAML::Node& semseg_cam_node,
			boost::shared_ptr<class carla::client::BlueprintLibrary> bp_library,	// NOLINT
			boost::shared_ptr<carla::client::Vehicle> vehicle,						// NOLINT
			const Eigen::Matrix4d& car_transform,
			bool log = true);
	void CaputreOnce();
	void Destroy();

	[[nodiscard]] size_t count() const;
	[[nodiscard]] bool waiting() const;
	[[nodiscard]] size_t depth_image_count() const;
	[[nodiscard]] size_t semantic_image_count() const;
	[[nodiscard]] std::tuple<bool, cv::Mat, cv::Mat, Eigen::Matrix4d> pop();
	[[nodiscard]] std::shared_ptr<geom::CameraGeometry> geometry() const;
private:
	bool save_depth_ = false;
	bool save_semantics_ = false;
	boost::shared_ptr<carla::client::Sensor> depth_sensor_;
	boost::shared_ptr<carla::client::Sensor> semantic_sensor_;
	std::shared_ptr<geom::CameraGeometry> geometry_;
	std::vector<cv::Mat> semantic_images_;
	std::vector<cv::Mat> depth_images_;
	std::vector<Eigen::Matrix4d> car_transforms_;
	std::mutex semantic_buffer_mutex_;
	std::mutex depth_buffer_mutex_;

	const Eigen::Matrix4d& car_transform_;
};



} // namespace data
#endif