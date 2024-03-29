#ifndef __HDF5_DATASET_H__
#define __HDF5_DATASET_H__

#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <tiff.h>

namespace statics{
constexpr static unsigned int front_rgb_channels = 4;
constexpr static unsigned int front_rgb_width = 640;
constexpr static unsigned int front_rgb_height = 480;
constexpr static unsigned int front_rgb_byte_count = front_rgb_height * front_rgb_width * front_rgb_channels;
constexpr static unsigned int top_semseg_width = 400;
constexpr static unsigned int top_semseg_height = 500;
constexpr static unsigned int top_semseg_byte_count = top_semseg_width * top_semseg_height;
constexpr static unsigned int transform_length = 16;
constexpr static unsigned int dataset_rank = 1;

constexpr static unsigned int open_mask = 0b0011;
constexpr static unsigned int dset_mask = 0b1100;
constexpr static unsigned int comp_type_mask = 0b1100000;
constexpr static unsigned int comp_lvl_mask  = 0b0011111;
} // namespace statics
struct MASSDataType {
    unsigned int agent_id;
    uint8 front_rgb[statics::front_rgb_channels * statics::front_rgb_height * statics::front_rgb_width];
    uint8 top_semseg[statics::top_semseg_height * statics::top_semseg_width];
    uint8 top_mask[statics::top_semseg_height * statics::top_semseg_width];
    double transform[statics::transform_length];
};
enum mode : unsigned int {
    FILE_RD_ONLY    = 0b0001,
    FILE_RDWR       = 0b0010,
    FILE_TRUNC      = 0b0011,
    DSET_OPEN       = 0b0100,
    DSET_CREAT      = 0b1000,
};
enum compression : unsigned int {
    NONE    = 0b0000000,
    ZLIB    = 0b0100000,
    SZIP    = 0b1000000
};

class HDF5Dataset
{
public:
    HDF5Dataset(const std::string& path, const std::string& dset_name, unsigned int flags,
                unsigned int compression, size_t init_size, size_t max_size, size_t chunk_size);
    HDF5Dataset(const HDF5Dataset&) = delete;
    HDF5Dataset& operator=(const HDF5Dataset&) = delete;
    HDF5Dataset& operator=(HDF5Dataset&&) = delete;
    HDF5Dataset(HDF5Dataset&&) = delete;
    ~HDF5Dataset() = default;

    void AppendElement(const MASSDataType* mass_data);
    void AddU32Attribute(uint32_t* attr_data, size_t attr_size, const std::string& attr_name);
    void ReadU32Attribute(const std::string& attr_name, uint32_t* buffer);
    void UpdateU32Attribute(const std::string& attr_name, uint32_t* buffer);
    MASSDataType ReadElement(size_t index) const;
    std::pair<hsize_t, hsize_t> GetCurrentSize() const;
    void Close();
private:
    void InitializeCompoundType();
    H5::DataSet dataset_;
    H5::CompType comp_type_;
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