#include "hdf5_api/hdf5_dataset.h"
#include <H5ArrayType.h>
#include <H5IntType.h>
#include <H5PredType.h>
#include <H5public.h>

HDF5Dataset::HDF5Dataset(const std::string& path, const std::string& dset_name, unsigned int flags, // NOLINT
                         unsigned int compression, size_t init_size, size_t max_size, size_t chunk_size,
                         unsigned int agent_count) {
    unsigned int hdf5_flags = 0;
    std::string file_action_str;
    switch (flags & statics::open_mask) {
        case FILE_RD_ONLY:
            hdf5_flags = H5F_ACC_RDONLY;
            file_action_str = "opened ";
            break;
        case FILE_RDWR:
            hdf5_flags = H5F_ACC_RDWR;
            file_action_str = "opened ";
            break;
        case FILE_TRUNC:
            hdf5_flags = H5F_ACC_TRUNC;
            file_action_str = "created ";
            break;
    }
    h5file_ = H5::H5File(path, hdf5_flags);
    std::cout << file_action_str << path << std::endl;
    InitializeCompoundType();
    switch (flags & statics::dset_mask) {
        case DSET_CREAT: {
            // creating the dataset
            hsize_t initial_size[] = {init_size};
            hsize_t maximum_size[] = {max_size * agent_count};
            H5::DataSpace data_space(statics::dataset_rank, initial_size, maximum_size);
            H5::DSetCreatPropList dset_params;
            hsize_t chunk_dims[] = {chunk_size};
            dset_params.setChunk(1, chunk_dims);
            switch (compression & statics::comp_type_mask) {
            case compression::ZLIB:
                dset_params.setDeflate(static_cast<int>(compression & statics::comp_lvl_mask));
                break;
            case compression::SZIP:
                dset_params.setSzip(H5_SZIP_NN_OPTION_MASK, compression & statics::comp_lvl_mask);
                break;
            };
            dataset_ = h5file_.createDataSet(dset_name, comp_type_, data_space, dset_params);
            write_index_ = 0;
            // creating the attribute
            hsize_t attr_dims[] = {1};
            unsigned int attr_data[] = {agent_count};
            H5::DataSpace attr_dataspace(1, attr_dims);
            H5::Attribute attribute = dataset_.createAttribute("agent_count", H5::PredType::STD_I32BE, attr_dataspace);
            attribute.write(H5::PredType::NATIVE_UINT32, attr_data);
            break;
        }
        case DSET_OPEN: {
            dataset_ = h5file_.openDataSet(dset_name);
            auto [dsize, dsize_max] = GetCurrentSize();
            write_index_ = dsize - 1;
            break;
        }
    }
    // hyperslab parameters
    _stride[0] = 1;
    _block[0] = 1;
    _count[0] = 1;
    // append parameter
    _mspace = H5::DataSpace (statics::dataset_rank, _count);
}

void HDF5Dataset::InitializeCompoundType() {
    // creating compound type
    // hsize_t name_dims[] = {statics::name_length};
    // H5::ArrayType name_array_type(H5::PredType::NATIVE_CHAR, 1, name_dims);
    hsize_t rgb_dims[] = {statics::front_rgb_channels * statics::front_rgb_height * statics::front_rgb_width};
    H5::ArrayType rgb_array_type(H5::PredType::NATIVE_UCHAR, 1, rgb_dims);
    hsize_t bev_dims[] = {statics::top_semseg_height * statics::top_semseg_width};
    H5::ArrayType bev_array_type(H5::PredType::NATIVE_UCHAR, 1, bev_dims);
    hsize_t tf_dims[] = {statics::transform_length};
    H5::ArrayType tf_array_type(H5::PredType::NATIVE_FLOAT, 1, tf_dims);
    comp_type_ = H5::CompType(sizeof(MASSDataType));
    comp_type_.insertMember("agent_id", HOFFSET(MASSDataType, agent_id), H5::PredType::NATIVE_UINT32); // NOLINT
    comp_type_.insertMember("front_rgb", HOFFSET(MASSDataType, front_rgb), rgb_array_type); // NOLINT
    comp_type_.insertMember("top_semseg", HOFFSET(MASSDataType, top_semseg), bev_array_type); // NOLINT
    comp_type_.insertMember("top_mask", HOFFSET(MASSDataType, top_mask), bev_array_type); // NOLINT
    comp_type_.insertMember("transform", HOFFSET(MASSDataType, transform), tf_array_type); // NOLINT
}

void HDF5Dataset::AppendElement(const MASSDataType* mass_data) {
    write_offset_[0] = {write_index_++};
    auto dspace = dataset_.getSpace();
    dspace.selectHyperslab(H5S_SELECT_SET, _count, write_offset_);
    // defining memory space (should match hyperslab)
    dataset_.write(mass_data, comp_type_, _mspace, dspace);
    // std::cout << "added element: " << write_index_ << std::endl;
    extension_[0] = write_index_ + 1;
    dataset_.extend(extension_);
}

void HDF5Dataset::Close() {
    dataset_.close();
    // std::cout << "dataset closed" << std::endl;
    h5file_.close();
    // std::cout << "file closed" << std::endl;
}

MASSDataType HDF5Dataset::ReadElement(size_t index) const {
    MASSDataType read_out{};
    auto dspace = dataset_.getSpace();
    // read_offset_[0] = {index};
    hsize_t count[] = {1};
    hsize_t read_offset[] = {index};
    dspace.selectHyperslab(H5S_SELECT_SET, count, read_offset);
    dataset_.read(&read_out, comp_type_, _mspace, dspace);
    return read_out;
}
std::pair<hsize_t, hsize_t> HDF5Dataset::GetCurrentSize() const {
    hsize_t dim[] = {0};
    hsize_t maxdim[] = {0};
    dataset_.getSpace().getSimpleExtentDims(dim, maxdim);
    return std::make_pair(dim[0], maxdim[0]);
}