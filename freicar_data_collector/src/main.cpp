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
    HDF5Dataset xtra("testfile2.hdf5", "testset", HDF5Dataset::FILE_TRUNC | HDF5Dataset::DSET_CREAT, 1, 20000, 32);
    for (size_t i = 0; i < 5; ++i) {
        MASSDataType data;
        sprintf(data.name, "d# %d", (int)i);
        data.transform[0] = 120 + i;
        xtra.AppendElement(&data);
        std::cout << "added element: " << i << std::endl;
    }
    size_t index = 10;
    // if (argc > 1) {
    //     index = atoi(argv[1]);
    // }
    for (size_t i = 0; i < index; ++i) {
        MASSDataType read_out = xtra.ReadElement(i);
        std::cout << read_out.name << ": " << read_out.transform[0] << std::endl;
    }
    auto [dset_size, dset_max_size] = xtra.GetCurrentSize();
    std::cout << dset_size << "/" << dset_max_size << std::endl;
    return 0;
}