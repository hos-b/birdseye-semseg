#ifndef __SURFACE_MAP_H__
#define __SURFACE_MAP_H__

#include <vector>

/*  
    this class divides the space into a 2D grid and fills each "pixel"
    with data generated from RGBD/D images. it performs the steps before
    generating a multi-level surface map, then uses heuristics to calculate
    a value for each pixel.
*/
namespace geom
{

template <typename PointType>
class SurfaceMap
{
public:
    struct Settings {
		float pixel_h;
		float pixel_w;
        float max_point_x;
		float max_point_y;
		unsigned int image_rows;
		unsigned int image_cols;
    };
    explicit SurfaceMap(SurfaceMap::Settings& settings) {
        cfg_ = settings;
        // pixel width & height
        grid_size_ = cfg_.image_rows * cfg_.image_cols;
        grid_.reserve(grid_size_ + 1);
        // TODO: check capacity
        #pragma omp parallel for
        for (size_t i = 0; i < grid_size_; ++i) {
            grid_.emplace_back(std::vector<PointType>());
        }
    }
    void Reserve(size_t new_size) {
        #pragma omp parallel for
        for (size_t i = 0; i < grid_size_; ++i) {
            grid_[i].reserve(grid_[i].size() + new_size);
        }
    }
    /* add a point to the surface map */
    void AddPoint(PointType pt) {
        float grid_x = (cfg_.max_point_y - pt.y) / cfg_.pixel_w;
        float grid_y = (cfg_.max_point_x - pt.x) / cfg_.pixel_h;
        if (grid_x < 0 || grid_y < 0 || grid_x > cfg_.image_cols || grid_y > cfg_.image_rows) {
            return;
        }
        size_t grid_index = static_cast<size_t>(grid_x * cfg_.image_cols + grid_y);
        grid_[grid_index].emplace_back(pt);
    }
    /* returns the points that belong to the  */
    std::vector<PointType>& GetGridPoints(size_t x, size_t y) {
        return grid_[x * cfg_.image_cols + y];
    }

    SurfaceMap(const SurfaceMap&) = delete;
    const SurfaceMap& operator=(const SurfaceMap&) = delete;
    SurfaceMap(SurfaceMap&&) = delete;
    const SurfaceMap& operator=(const SurfaceMap&&) = delete;

    ~SurfaceMap() {
        grid_.clear();
    }

private:
    Settings cfg_;
    size_t grid_size_;
    std::vector<std::vector<PointType>> grid_;
};

} // namespace geom
#endif