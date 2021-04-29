#ifndef __SURFACE_MAP_H__
#define __SURFACE_MAP_H__

#include <vector>
#include <array>
#include <limits>

/*  
    this class divides the space into a 2D grid and fills each "pixel"
    with data generated from RGBD/D images. it performs the steps before
    generating a multi-level surface map, then uses heuristics to calculate
    a value for each pixel. Dynamic memory allocation turned out to be the
    wrong way to go.
*/
namespace geom
{

struct SurfaceMapSettings {
    float pixel_h;
    float pixel_w;
    float max_point_x;
    float max_point_y;
    unsigned int image_rows;
    unsigned int image_cols;
};

enum class OverflowBehavior {
    FIFO, // first in, first overwritten
    IGNORE, // do not add if pixel container is full
    SPACED_REPLACE // on avg every 10th point gets replaced
};

/*
    underlying template for surface map that allows specialization for
    different ways of handling large amounts of points
*/
template <typename PointType, OverflowBehavior OVF, uint16_t PX_CAP>
class overflowable {
};


template <typename PointType, uint16_t PX_CAP>
class overflowable<PointType, OverflowBehavior::IGNORE, PX_CAP> {
public:
    void InitializeContainers() {
        // pixel width & height
        size_t grid_size = cfg_.image_rows * cfg_.image_cols;
        grid_.reserve(grid_size);
        for (size_t i = 0; i < grid_size; ++i) {
            grid_.emplace_back(std::vector<pcl::PointXYZL>());
            grid_.back().reserve(PX_CAP);
        }
    }
    /* add a point to the surface map */
    void AddPoint(PointType pt) {
        int64_t grid_x = static_cast<int64_t>((cfg_.max_point_y - pt.y) / cfg_.pixel_w);
        int64_t grid_y = static_cast<int64_t>((cfg_.max_point_x - pt.x) / cfg_.pixel_h);
        // skip if outside perception boundary
        if (grid_x < 0 || grid_y < 0 || grid_x >= cfg_.image_cols || grid_y >= cfg_.image_rows) {
            return;
        }
        size_t index = grid_y * cfg_.image_cols + grid_x;
        // skip if pixel container is full
        if (grid_[index].size() == PX_CAP) {
            return;
        }
        grid_[index].emplace_back(pt);
    }
protected:
    std::vector<std::vector<PointType>> grid_;
    SurfaceMapSettings cfg_;
};

template <typename PointType, uint16_t PX_CAP>
class overflowable<PointType, OverflowBehavior::FIFO, PX_CAP> {
public:
    void InitializeContainers() {
        // pixel width & height   
        size_t grid_size = cfg_.image_rows * cfg_.image_cols;
        grid_.reserve(grid_size);
        for (size_t i = 0; i < grid_size; ++i) {
            grid_.emplace_back(std::vector<pcl::PointXYZL>());
            grid_.back().reserve(PX_CAP);
        }
        overflow_indices_.insert(grid_.end(), grid_size, 0);
        static_assert(PX_CAP <= std::numeric_limits<uint16_t>::max()); // don't use super computers
    }
    /* add a point to the surface map */
    void AddPoint(PointType pt) {
        int64_t grid_x = static_cast<int64_t>((cfg_.max_point_y - pt.y) / cfg_.pixel_w);
        int64_t grid_y = static_cast<int64_t>((cfg_.max_point_x - pt.x) / cfg_.pixel_h);
        // skip if outside perception boundary
        if (grid_x < 0 || grid_y < 0 || grid_x >= cfg_.image_cols || grid_y >= cfg_.image_rows) {
            return;
        }
        size_t index = grid_y * cfg_.image_cols + grid_x;
        // replace if pixel container is full
        if (grid_[index].size() == PX_CAP) {
            auto& overflow_index = overflow_indices_[index];
            grid_[index][overflow_index] = pt;
            overflow_index = (overflow_index + 1) % PX_CAP;
        } else {
            grid_[index].emplace_back(pt);
        }
    }
protected:
    std::vector<std::vector<PointType>> grid_;
    std::vector<uint16_t> overflow_indices_;
    SurfaceMapSettings cfg_;
};

template <typename PointType, uint16_t PX_CAP>
class overflowable<PointType, OverflowBehavior::SPACED_REPLACE, PX_CAP> {
public:
    void InitializeContainers() {
        // pixel width & height
        size_t grid_size = cfg_.image_rows * cfg_.image_cols;
        grid_.reserve(grid_size);
        for (size_t i = 0; i < grid_size; ++i) {
            grid_.emplace_back(std::vector<pcl::PointXYZL>());
            grid_.back().reserve(PX_CAP);
        }
        overflow_indices_.insert(grid_.end(), grid_size, 0);
        static_assert(PX_CAP <= std::numeric_limits<uint16_t>::max()); // don't use super computers
        static_assert(PX_CAP % index_jump_ != 0); // somehow change index jump
    }
    /* add a point to the surface map */
    void AddPoint(PointType pt) {
        int64_t grid_x = static_cast<int64_t>((cfg_.max_point_y - pt.y) / cfg_.pixel_w);
        int64_t grid_y = static_cast<int64_t>((cfg_.max_point_x - pt.x) / cfg_.pixel_h);
        // skip if outside perception boundary
        if (grid_x < 0 || grid_y < 0 || grid_x >= cfg_.image_cols || grid_y >= cfg_.image_rows) {
            return;
        }
        size_t index = grid_y * cfg_.image_cols + grid_x;
        // skip if pixel container is full
        if (grid_[index].size() == PX_CAP) {
            auto& overflow_index = overflow_indices_[index];
            grid_[index][overflow_index] = pt;
            overflow_index = (overflow_index + index_jump_) % PX_CAP;
        } else {
            grid_[index].emplace_back(pt);
        }
    }
protected:
    std::vector<std::vector<PointType>> grid_;
    std::vector<uint16_t> overflow_indices_;
    static constexpr size_t index_jump_ = (PX_CAP / 10) + 1;
    SurfaceMapSettings cfg_;
};

template <typename PointType, OverflowBehavior OVF, uint16_t PX_CAP>
class SurfaceMap : public overflowable<PointType, OVF, PX_CAP>
{
public:
    explicit SurfaceMap(SurfaceMapSettings& settings) {
        this->cfg_ = settings;
        this->InitializeContainers();
    }
    /* returns the points that belong to the given pixel in the image frame */
    const std::vector<PointType>& GetGridPoints(size_t col, size_t row) const {
        return this->grid_[row * this->cfg_.image_cols + col];
    }

    SurfaceMap(const SurfaceMap&) = delete;
    const SurfaceMap& operator=(const SurfaceMap&) = delete;
    SurfaceMap(SurfaceMap&&) = delete;
    const SurfaceMap& operator=(const SurfaceMap&&) = delete;

    ~SurfaceMap() {
        this->grid_.clear();
    }

};

} // namespace geom
#endif