#include "geometry/pointcloud.h"
#include <open3d/geometry/PointCloud.h>


namespace geom
{
SemanticCloud::SemanticCloud() {
    std::cout << "hello" << std::endl;
    open3d::PrintOpen3DVersion();
}

} // namespace geom