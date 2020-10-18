#ifndef __HDF5_DATASET_H__
#define __HDF5_DATASET_H__

#include <iostream>
#include <string>
#include <H5Cpp.h>

namespace statics{
constexpr static unsigned int name_length = 10;
constexpr static unsigned int front_rgb_channels = 3;
constexpr static unsigned int front_rgb_width = 640;
constexpr static unsigned int front_rgb_height = 480;
constexpr static unsigned int top_semseg_width = 1000;
constexpr static unsigned int top_semseg_height = 1000;
constexpr static unsigned int top_depth_width = 1000;
constexpr static unsigned int top_depth_height = 1000;
constexpr static unsigned int transform_length = 16;
constexpr static unsigned int dataset_rank = 1;
}
struct MASSDataType {
    char name[statics::name_length];
    unsigned char front_rgb[statics::front_rgb_channels][statics::front_rgb_height][statics::front_rgb_width];
    unsigned short top_semseg[statics::top_semseg_height][statics::top_semseg_width];
    float top_depth[statics::top_depth_height][statics::top_depth_width];
    float transform[statics::transform_length];
};

class HDF5Dataset
{
public:
    HDF5Dataset(const std::string& path, const std::string& dset_name, unsigned int flags, size_t init_size, size_t max_size, size_t chunk_size);
    HDF5Dataset(const HDF5Dataset&) = delete;
    HDF5Dataset(HDF5Dataset&&) = delete;
    void AppendElement(const MASSDataType* mass_data);
    MASSDataType ReadElement(size_t index);
    void Close();
    H5::DataSet dataset_;
    H5::CompType comp_type_;
private:
    void InitializeCompoundType();
    size_t write_index_;
    H5::H5File h5file_;
    H5::DataSpace _mspace;

    // hyperslab selection
    hsize_t _count[1];
    hsize_t _stride[1];
    hsize_t _block[1];
    // adding
    hsize_t extension_[1];
    hsize_t write_offset_[1];
    hsize_t read_offset_[1];
};

#endif