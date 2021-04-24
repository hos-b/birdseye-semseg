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
        float max_point_x;
		float min_point_x;
		float max_point_y;
		float min_point_y;
		unsigned int image_rows;
		unsigned int image_cols;
    };
    explicit SurfaceMap(SurfaceMap::Settings& settings, size_t points_per_pixel) {
        cfg_ = settings;
        // pixel width & height
        pixel_w_ = (cfg_.max_point_y - cfg_.min_point_y;) / static_cast<double>(cfg_.image_cols);
	    pixel_h_ = (cfg_.max_point_x - cfg_.min_point_x;) / static_cast<double>(cfg_.image_rows);
        size_t grid_size = cfg_.image_rows * cfg_.image_cols;
        for (size_t i = 0; i < grid_size_; ++i) {
            grid_.emplace_back(std::vector<PointType>());
            grid_.back().reserve(points_per_pixel);
        }
    }
    /* add a point to the surface map */
    void AddPoint(PointType pt) {
        float grid_x = (cfg_.max_point_y - pt.y) / pixel_w_;
        float grid_y = (cfg_.max_point_x - pt.x) / pixel_h_;
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
    std::vector<std::vector<PointType>> grid_;
};

} // namespace geom
#endif