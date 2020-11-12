#include "mass_agent/sensors.h"
#include "geometry/camera_geomtry.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

#define cam_log(x) if (log) std::cout << x << std::endl;

namespace cc = carla::client;
namespace cg = carla::geom;
namespace data
{

RGBCamera::RGBCamera(const YAML::Node& rgb_cam_node,
            boost::shared_ptr<class carla::client::BlueprintLibrary> bp_library,	// NOLINT
            boost::shared_ptr<carla::client::Vehicle> vehicle,					    // NOLINT
            bool log) {
    cam_log("setting up rgb camera");
    auto cam_blueprint = *bp_library->Find(rgb_cam_node["type"].as<std::string>());
    // setting camera attributes from the yaml
    for (YAML::const_iterator it = rgb_cam_node.begin(); it != rgb_cam_node.end(); ++it) {
        auto key = it->first.as<std::string>();
        if (cam_blueprint.ContainsAttribute(key)) {
            cam_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
    }
    // spawn the camera attached to the vehicle
    auto camera_transform = cg::Transform{
        cg::Location{rgb_cam_node["x"].as<float>(),
                    rgb_cam_node["y"].as<float>(),
                    rgb_cam_node["z"].as<float>()},
        cg::Rotation{rgb_cam_node["pitch"].as<float>(),
                    rgb_cam_node["yaw"].as<float>(),
                    rgb_cam_node["roll"].as<float>()}};
    auto generic_actor = vehicle->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle.get());
    sensor_ = boost::static_pointer_cast<cc::Sensor>(generic_actor);
    geometry_ = std::make_unique<geom::CameraGeometry>(sensor_,
                                                       rgb_cam_node["fov"].as<float>(),
                                                       rgb_cam_node["image_size_x"].as<unsigned int>(),
                                                       rgb_cam_node["image_size_y"].as<unsigned int>());
    // register a callback to publish images
    sensor_->Listen([this](const boost::shared_ptr<carla::sensor::SensorData>& data) {
        std::lock_guard<std::mutex> guard(buffer_mutex_);
        auto image = boost::static_pointer_cast<csd::Image>(data);
        auto mat = cv::Mat(image->GetHeight(), image->GetWidth(), CV_8UC4, image->data());
        if (save_) {
            images_.emplace_back(mat.clone());
            save_ = false;
            std::cout << "saving rgb:" << images_.size() << std::endl;
            // cv::imwrite("/home/hosein/catkin_ws/src/mass_data_collector/guide/rgb.png", images_[0]);
        }
    });
}
/* lets the camera save the next frame */
void RGBCamera::CaputreOnce() {
    save_ = true;
}
/* destroys the sensor and clears the queue */
void RGBCamera::Destroy() {
    if (sensor_) {
        sensor_->Destroy();
    }
    images_.clear();
}
/* returns the number of images currently stored in buffer */
size_t RGBCamera::count() {
    return images_.size();
}
/* returns true and the oldest buffer element. false and empty element if empty */
std::pair <bool, cv::Mat> RGBCamera::pop() {
    cv::Mat oldest;
    bool empty = images_.empty();
    if (!empty) {
        std::lock_guard<std::mutex> guard(buffer_mutex_);
        oldest = images_.front();
        images_.erase(images_.begin());
    }
    return std::make_pair(empty, oldest);
}
/* returns the camera geometry */
std::shared_ptr<geom::CameraGeometry> RGBCamera::geometry() {
    return geometry_;
}
// ------------------------ SemnaticPointCloudCamera -----------------------------
SemanticPointCloudCamera::SemanticPointCloudCamera(const YAML::Node& depth_cam_node,
            const YAML::Node& semseg_cam_node,
			boost::shared_ptr<class carla::client::BlueprintLibrary> bp_library,	// NOLINT
			boost::shared_ptr<carla::client::Vehicle> vehicle,						// NOLINT
			bool log) {
    cam_log ("setting up top depth camera");
    // usual camera info stuff
    auto dcam_blueprint = *bp_library->Find(depth_cam_node["type"].as<std::string>());
    // setting camera attributes from the yaml
    for (YAML::const_iterator it = depth_cam_node.begin(); it != depth_cam_node.end(); ++it) {
        auto key = it->first.as<std::string>();
        if (dcam_blueprint.ContainsAttribute(key)) {
            dcam_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
    }
    // spawn the depth camera attached to the vehicle
    auto camera_transform = cg::Transform{
        cg::Location{depth_cam_node["x"].as<float>(),
                        depth_cam_node["y"].as<float>(),
                        depth_cam_node["z"].as<float>()},
        cg::Rotation{depth_cam_node["pitch"].as<float>(),
                        depth_cam_node["yaw"].as<float>(),
                        depth_cam_node["roll"].as<float>()}};
    auto generic_actor = vehicle->GetWorld().SpawnActor(dcam_blueprint, camera_transform, vehicle.get());
    depth_sensor_ = boost::static_pointer_cast<cc::Sensor>(generic_actor);
    // register a callback to publish images
    depth_sensor_->Listen([this](const boost::shared_ptr<carla::sensor::SensorData>& data) {
        auto image = boost::static_pointer_cast<csd::Image>(data);
        if (save_depth_) {
            std::lock_guard<std::mutex> guard(depth_buffer_mutex_);
            depth_images_.emplace_back(DecodeToDepthMat(image).clone());
            // cv::exp(depth_images_[0], depth_images_[0]); good for debugging
            save_depth_ = false;
            std::cout << "saving depth:" << depth_images_.size() << std::endl;
            // cv::imwrite("/home/hosein/catkin_ws/src/mass_data_collector/guide/depth.png", depth_images_[0]);
        }
    });
    cam_log("setting up top semantic camera");
    auto scam_blueprint = *bp_library->Find(semseg_cam_node["type"].as<std::string>());
    // setting camera attributes from the yaml
    for (YAML::const_iterator it = semseg_cam_node.begin(); it != semseg_cam_node.end(); ++it) {
        auto key = it->first.as<std::string>();
        if (scam_blueprint.ContainsAttribute(key)) {
            scam_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
    }
    // spawn the camera attached to the vehicle.	
    camera_transform = cg::Transform{
        cg::Location{semseg_cam_node["x"].as<float>(),
                    semseg_cam_node["y"].as<float>(),
                    semseg_cam_node["z"].as<float>()},
        cg::Rotation{semseg_cam_node["pitch"].as<float>(),
                    semseg_cam_node["yaw"].as<float>(),
                    semseg_cam_node["roll"].as<float>()}};
    generic_actor = vehicle->GetWorld().SpawnActor(scam_blueprint, camera_transform, vehicle.get());
    semantic_sensor_ = boost::static_pointer_cast<cc::Sensor>(generic_actor);
    geometry_ = std::make_unique<geom::CameraGeometry>(semantic_sensor_,
                                                       semseg_cam_node["fov"].as<float>(),
                                                       semseg_cam_node["image_size_x"].as<unsigned int>(),
                                                       semseg_cam_node["image_size_y"].as<unsigned int>());
    // callback
    semantic_sensor_->Listen([this](const boost::shared_ptr<carla::sensor::SensorData>& data) {
        auto image = boost::static_pointer_cast<csd::Image>(data);
        if (save_semantics_) {
            std::lock_guard<std::mutex> guard(semantic_buffer_mutex_);
            semantic_images_.emplace_back(DecodeToCityScapesPalleteSemSegMat(image));
            save_semantics_ = false;
            std::cout << "saving semantics:" << semantic_images_.size() << semantic_images_.back().cols
                                                                 << "x" << semantic_images_.back().rows << std::endl;
            // cv::imwrite("/home/hosein/catkin_ws/src/mass_data_collector/guide/semantic.png", semantic_images_[0]);
        }
    });
}
/* destroys the sensors */
void SemanticPointCloudCamera::Destroy() {
    if (depth_sensor_) {
        depth_sensor_->Destroy();
    }
    if (semantic_sensor_) {
        semantic_sensor_->Destroy();
    }
    semantic_images_.clear();
    depth_images_.clear();
}
/* allows the cameras to save the very next frame */
void SemanticPointCloudCamera::CaputreOnce() {
    save_semantics_ = true;
    save_depth_ = true;
}
/* returns the minimum size of the two buffer */
size_t SemanticPointCloudCamera::count() {
    return std::min(semantic_images_.size(), depth_images_.size());
}
/* returns depth buffer size */
size_t SemanticPointCloudCamera::depth_image_count() {
    return depth_images_.size();
}
/* returns semantic buffer size */
size_t SemanticPointCloudCamera::semantic_image_count() {
    return semantic_images_.size();
}
/* returns true and oldest semnatic|depth images. false and two empty images if empty */
std::tuple<bool, cv::Mat, cv::Mat> SemanticPointCloudCamera::pop() {
    cv::Mat semantic;
    cv::Mat depth;
    bool success = std::min(semantic_images_.size(), depth_images_.size()) > 0;
    if (success) {
        std::lock(semantic_buffer_mutex_, depth_buffer_mutex_);
        std::lock_guard<std::mutex> sguard(semantic_buffer_mutex_, std::adopt_lock);
        std::lock_guard<std::mutex> dguard(depth_buffer_mutex_, std::adopt_lock);
        semantic = std::move(semantic_images_.front());
        depth = std::move(depth_images_.front());
        semantic_images_.erase(semantic_images_.begin());
        depth_images_.erase(depth_images_.begin());
    }
    return std::make_tuple(success, semantic, depth);
}
/* returns the camera geometry */
std::shared_ptr<geom::CameraGeometry> SemanticPointCloudCamera::geometry() {
    return geometry_;
}
// custom conversion functions

/* converts an rgb coded matrix into an OpenCV mat containg real depth values
   returns CV_32FC1
*/
cv::Mat DecodeToDepthMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image) { // NOLINT
	auto color_coded_depth = cv::Mat(carla_image->GetHeight(),
									 carla_image->GetWidth(),
									 CV_8UC4,
									 carla_image->data());
	std::vector<cv::Mat> splits;
	cv::split(color_coded_depth, splits);
	cv::Mat real_depth(color_coded_depth.rows, color_coded_depth.cols, CV_32FC1);
	cv::Mat B;
	cv::Mat G;
	cv::Mat R;
	splits[0].convertTo(B, CV_32FC1);
	splits[1].convertTo(G, CV_32FC1);
	splits[2].convertTo(R, CV_32FC1);
	// carla's color coded depth to real depth conversion
	real_depth = ((R + G * 256.0f + B * (256.0f * 256.0f)) / (256.0f * 256.0f * 256.0f - 1.0f)) * 1000.0f; // NOLINT
	return real_depth.clone();
}
/* converts an rgb coded depth image into an OpenCV mat containg logarithmic depth values (good for visualization)
   returns CV_8UC1
*/
cv::Mat DecodeToLogarithmicDepthMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image) { // NOLINT
	auto image_view = carla::image::ImageView::MakeColorConvertedView(carla::image::ImageView::MakeView(*carla_image),
																	  carla::image::ColorConverter::LogarithmicDepth());
	static_assert(sizeof(decltype(image_view)::value_type) == 1, "single channel");
	decltype(image_view)::value_type raw_data[image_view.width() * image_view.height()];
	boost::gil::copy_pixels(image_view, boost::gil::interleaved_view(image_view.width(),
																	 image_view.height(),
																	 raw_data,
																	 image_view.width()));
	// if not clone, corrupted stack memory will be used
	return cv::Mat(image_view.height(), image_view.width(), CV_8UC1, raw_data).clone();
}
/* extractss the red channel of the BGRA image which contains semantic data
   returns CV_8UC1
 */
cv::Mat DecodeToSemSegMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image) { // NOLINT
	auto semseg_mat_4c = cv::Mat(carla_image->GetHeight(), carla_image->GetWidth(), CV_8UC4, carla_image->data());
	std::vector<cv::Mat> splits;
	cv::split(semseg_mat_4c, splits);
	return splits[2].clone(); // BG->R<-A
}
/* converts the semantic data to an RGB image with cityscapes' semantic pallete
   returns CV_8UC3
*/
cv::Mat DecodeToCityScapesPalleteSemSegMat(boost::shared_ptr<csd::ImageTmpl<csd::Color>> carla_image) { // NOLINT
	// TODO(hosein): do the pallete yourself with opencv to avoid double copy
	auto image_view = carla::image::ImageView::MakeColorConvertedView(carla::image::ImageView::MakeView(*carla_image), 
																	  carla::image::ColorConverter::CityScapesPalette());
	auto rgb_view = boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(image_view);
	using pixel = decltype(rgb_view)::value_type;
	static_assert(sizeof(pixel) == 3, "RGB");
	pixel raw_data[rgb_view.width() * rgb_view.height()];
	boost::gil::copy_pixels(rgb_view, boost::gil::interleaved_view(rgb_view.width(),
																   rgb_view.height(),
																   raw_data,
																   rgb_view.width() * sizeof(pixel)));
	// if not clone, corrupted stack memory will be used
	return cv::Mat(image_view.height(), image_view.width(), CV_8UC3, raw_data).clone();
}

} // namespace data

#undef cam_log
