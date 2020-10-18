#include <iostream>
#include <string>
#include <cstring>
// #include "hdf5/openmpi/hdf5.h"
// #include "hdf5/serial/H5Cpp.h"
#include "hdf5_dataset.h"

#define DATASET_RANK 1
struct TType {
    char name[10];
    float transform[16];
};

int main(int argc, char** argv)
{
    std::cout << "starting..." << std::endl;
    H5::H5File h5file("testfile.hdf5", H5F_ACC_TRUNC);
    // creating compound type
    hsize_t name_dims[] = {10};
    H5::ArrayType name_array_type(H5::PredType::NATIVE_CHAR, 1, name_dims);
    // hsize_t frgb_dims[] = {3, 640, 480};
    // H5::ArrayType frgb_array_type(H5::PredType::NATIVE_UCHAR, 3, frgb_dims);
    // hsize_t tss_dims[] = {1000, 1000};
    // H5::ArrayType tss_array_type(H5::PredType::NATIVE_USHORT, 2, tss_dims);
    // hsize_t tdepth_dims[] = {1000, 1000};
    // H5::ArrayType tdepth_array_type(H5::PredType::NATIVE_FLOAT, 2, tdepth_dims);
    hsize_t tf_dims[] = {16};
    H5::ArrayType tf_array_type(H5::PredType::NATIVE_FLOAT, 1, tf_dims);
    H5::CompType mass_type(sizeof(TType));
    mass_type.insertMember("name", HOFFSET(TType, name), name_array_type);
    // mass_type.insertMember("front_rgb", HOFFSET(TType, front_rgb), frgb_array_type);
    // mass_type.insertMember("top_semseg", HOFFSET(TType, top_semseg), tss_array_type);
    // mass_type.insertMember("top_depth", HOFFSET(TType, top_depth), tdepth_array_type);
    mass_type.insertMember("transform", HOFFSET(TType, transform), tf_array_type);
    std::cout << "created mass type: " << sizeof(TType) << std::endl;
    // // dataset size (initial, max)
    hsize_t initial_size[] = {1};
    hsize_t maximum_size[] = {20000};
    H5::DataSpace dspace(DATASET_RANK, initial_size, maximum_size);
    // filler value
    // MASSDataType filler;
    // strcpy(filler.name, "filler");
    // filler.transform[0] = filler.transform[15] = -1;
    // dset_params.setFillValue(mass_type, &filler);
    // chunk size
    H5::DSetCreatPropList dset_params;
    hsize_t chunk_dims[] = {32};
    dset_params.setChunk(1, chunk_dims);
    // // creating dataset
    H5::DataSet dataset = h5file.createDataSet("testset", mass_type, dspace, dset_params);
    // HDF5Dataset dset("testfile.hdft", "testset", H5F_ACC_TRUNC, 1, 20000, 32);
    std::cout << "created dataset" << std::endl;
    for (size_t i = 0; i < 15; ++i) {
        TType data;
        strcpy(data.name, "testing");
        data.transform[0] = 120 + i;
        hsize_t write_offset[1] = {i};
        hsize_t write_count[1] = {1};
        hsize_t write_stride[1] = {1};
        hsize_t write_block[1] = {1};
        auto wdspace = dataset.getSpace();
        wdspace.selectHyperslab(H5S_SELECT_SET, write_count, write_offset);
        // defining memory space (should match hyperslab)
        H5::DataSpace mspace(statics::dataset_rank, write_count);
        dataset.write(&data, mass_type, mspace, wdspace);
        std::cout << "added element: " << i << std::endl;
        hsize_t extension[] = {i + 2};
        dataset.extend(extension);
    }
    size_t index = 0;
    if (argc > 1) {
        index = atoi(argv[1]);
    }
    TType read_out;
    auto rdspace = dataset.getSpace();
    hsize_t dims_out[1];
    hsize_t f_offset[1] = {index};
    hsize_t read_count[1] = {1};
    rdspace.selectHyperslab(H5S_SELECT_SET, read_count, f_offset);
    std::cout << "got dspace hyperslab" << std::endl;
    // hsize_t m_offset[1] = {0};
    H5::DataSpace mspace(statics::dataset_rank, dims_out);
    // mspace.selectHyperslab(H5S_SELECT_SET, read_count, m_offset);
    std::cout << "got mspace hyperslab" << std::endl;
    dataset.read(&read_out, mass_type, mspace, rdspace);
    std::cout << read_out.name << ": " << read_out.transform[0] << std::endl;
    
    return 0;
}