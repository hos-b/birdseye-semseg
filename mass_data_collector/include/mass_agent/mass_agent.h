#ifndef __AISCAR_RL_AGENT__
#define __AISCAR_RL_AGENT__

#include <string>
#include <memory>
#include <thread>

#include "carla/client/Walker.h"
#include "carla/client/Waypoint.h"
#include "carla/geom/Location.h"
#include "mass_agent/sensors.h"
#include "geometry/semantic_cloud.h"

// Eigen stuff
#include <Eigen/Dense>

// OpenCV stuff
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// CARLA stuff
#include <carla/client/Map.h>
#include <carla/client/World.h>

#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>
#include <vector>

namespace cc = carla::client;
namespace agent {
class MassAgent;
using WaypointKDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, MassAgent>,
														   MassAgent, 3>;

class MassAgent
{
public:
	~MassAgent();
	explicit MassAgent();
	MassAgent(const MassAgent&) = delete;
	const MassAgent& operator=(const MassAgent&) = delete;
	MassAgent(MassAgent&&) = delete;
	const MassAgent& operator=(MassAgent&&) = delete;
	void GenerateDataPoint();
	void CaptureOnce(bool log = true);
	boost::shared_ptr<carla::client::Waypoint> SetRandomPose();
	boost::shared_ptr<carla::client::Waypoint> SetRandomPose(boost::shared_ptr<carla::client::Waypoint> initial_wp);
	void WriteMapToFile(const std::string& path);


	[[nodiscard]] inline double carla_x() const;
	[[nodiscard]] inline double carla_y() const;
	[[nodiscard]] inline double carla_z() const;
	
	// obligatory kd-tree stuff
	[[nodiscard]] inline size_t kdtree_get_point_count() const { return kd_points().size(); } // NOLINT
	[[nodiscard]] inline float kdtree_get_pt(const size_t idx, const size_t dim) const { // NOLINT
		if (dim == 0) {
			return kd_points()[idx]->GetTransform().location.x;
		}
		if (dim == 1) {
			return kd_points()[idx]->GetTransform().location.y;
		}
		return kd_points()[idx]->GetTransform().location.z;
	}
	template<class BBox>
	bool kdtree_get_bbox(BBox& /* bb */) const {return false;}
private:
	[[nodiscard]] std::tuple<float, float, float> GetPostion() const;
	void SetupSensors();
	void DestroyAgent();
	void InitializeKDTree();

	static std::vector<std::string> GetBlueprintNames();
	static std::vector<const MassAgent*>& agents();
	static std::vector<boost::shared_ptr<carla::client::Waypoint>>& kd_points();
	static std::unique_ptr<cc::Client>& carla_client();
	// state
	uint16 id_;
	Eigen::Matrix4d transform_;
	std::string blueprint_name_;
	double width_, length_;
	// data
	std::vector<Eigen::Matrix4d> datapoint_transforms_;
	cv::Mat vehicle_mask_;
	// carla stuff
	boost::shared_ptr<carla::client::Vehicle> vehicle_;
	// carla sensors
	std::unique_ptr<data::RGBCamera> front_cam_;
	std::vector<std::unique_ptr<data::SemanticPointCloudCamera>> semantic_pc_cams_;
	// lookup containers
	std::unique_ptr<WaypointKDTree> kd_tree_;
};

} // namespace agent
#endif