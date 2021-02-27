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

// HDF5 stuff
#include "hdf5_api/hdf5_dataset.h"

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

	boost::shared_ptr<carla::client::Waypoint> SetRandomPose(const std::unordered_map<int, bool>& restricted_roads);
	boost::shared_ptr<carla::client::Waypoint>
	SetRandomPose(boost::shared_ptr<carla::client::Waypoint> initial_wp,
				  size_t knn_pts, const MassAgent* agents, const bool* deadlock,
				  std::vector<unsigned int> indices, unsigned int max_index,
				  const std::unordered_map<int, bool>& restricted_roads);
	void HideAgent();
	// TODO: implement
	void RandomizeColor();
	void PauseSensorCallbacks();
	void ResumeSensorCallbacks();

	MASSDataType GenerateDataPoint(unsigned int agent_batch_index);
	// transform related
	inline double carla_x() const;
	inline double carla_y() const;
	inline double carla_z() const;
	inline Eigen::Matrix4d transform() {return transform_;}
	
	// static stuff
	static std::unique_ptr<cc::Client>& carla_client();
	static std::vector<const MassAgent*>& agents();
	static void DebugMultiAgentCloud(MassAgent* agents, size_t size, const std::string& path);

	// debug functions
	cv::Mat GetMap();
	void ExploreMap();

	// obligatory kd-tree stuff
	inline size_t kdtree_get_point_count() const { return kd_points_.size(); }
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
		if (dim == 0) {
			return kd_points_[idx]->GetTransform().location.x;
		}
		if (dim == 1) {
			return kd_points_[idx]->GetTransform().location.y;
		}
		return kd_points_[idx]->GetTransform().location.z;
	}
	template<class BBox>
	bool kdtree_get_bbox(BBox& /* bb */) const {return false;}


private:
	std::tuple<float, float, float> GetPostion() const;
	void SetupSensors(float rgb_cam_shift);
	void DestroyAgent();
	void CaptureOnce();
	void InitializeKDTree();
	void AssertSize(size_t size);
	static std::vector<std::string> GetBlueprintNames();
	static geom::SemanticCloud::Settings& sc_settings();


	// state
	uint16 id_;
	Eigen::Matrix4d transform_;
	std::string blueprint_name_;
	double vehicle_width_, vehicle_length_;
	// carla stuff
	boost::shared_ptr<carla::client::Vehicle> vehicle_;
	std::vector<boost::shared_ptr<carla::client::Waypoint>> kd_points_;
	// carla sensors
	std::unique_ptr<data::RGBCamera> front_rgb_;
	std::unique_ptr<data::SemanticPointCloudCamera> front_mask_pc_;
	std::vector<std::unique_ptr<data::SemanticPointCloudCamera>> semantic_pc_cams_;
	// lookup containers
	std::unique_ptr<WaypointKDTree> kd_tree_;
};

} // namespace agent
#endif