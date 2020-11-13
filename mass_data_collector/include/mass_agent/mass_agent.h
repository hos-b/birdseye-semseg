#ifndef __AISCAR_RL_AGENT__
#define __AISCAR_RL_AGENT__

#include <string>
#include <memory>
#include <thread>

#include "map_core/freicar_map.h"
#include "mass_agent/sensors.h"
#include "freicar_map/thrift_map_proxy.h"
#include "geometry/semantic_cloud.h"

// Eigen stuff
#include <Eigen/Dense>

// OpenCV stuff
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// CARLA stuff
#include <carla/client/Map.h>
#include <carla/client/World.h>

#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>
#include <vector>

// Open3D stuff ?

namespace cc = carla::client;


namespace agent {

class MassAgent
{
public:
	~MassAgent();
	MassAgent(const MassAgent&) = delete;
	const MassAgent& operator=(const MassAgent&) = delete;
	MassAgent(MassAgent&&) = delete;
	const MassAgent& operator=(MassAgent&&) = delete;
	explicit MassAgent();
	void ActivateCarlaAgent(const std::string &address, unsigned int port);
	void GenerateDataPoint();

	[[nodiscard]] bool SetRandomPose();
	[[nodiscard]] const Eigen::Matrix4d& GetTransformReference() const;

	[[nodiscard]] inline double x() const;
	[[nodiscard]] inline double y() const;
	[[nodiscard]] inline double z() const;
	
private:
	[[nodiscard]] std::tuple<float, float, float> GetPostion() const;
	void SetupSensors();
	void DestroyAgent();
	static std::vector<const MassAgent*>& agents();
	// state
	freicar::map::Map& map_instance_;
	uint16 id_;
	Eigen::Matrix4d transform_;
	geom::SemanticCloud semantic_cloud_;
	// carla stuff
	std::unique_ptr<cc::Client> carla_client_;
	boost::shared_ptr<carla::client::Vehicle> vehicle_;

	// carla sensors
	std::unique_ptr<data::RGBCamera> front_cam_;
	std::unique_ptr<data::SemanticPointCloudCamera> semantic_pc_cam_center_;	
};

} // namespace agent
#endif