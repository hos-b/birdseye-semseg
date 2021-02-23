#include <iostream>
#include <string>
#include <cstring>
#include "hdf5_api/hdf5_dataset.h"

#define DATASET_RANK 1
struct TType {
    char name[10];
    float transform[16];
};

int main()
{
    std::cout << "starting..." << std::endl;
    HDF5Dataset xtra("testfile2.hdf5", "testset", mode::FILE_TRUNC | mode::DSET_CREAT,
                                                 compression::ZLIB | 6, 1, 20000, 32);
    size_t index = 100;
    // writing
    for (size_t i = 0; i < index; ++i) {
        MASSDataType data{};
        data.agent_id = i;
        data.front_rgb[0] = i + 1;
        data.top_mask[0] = i + 2;
        data.top_semseg[0] = i + 3;
        xtra.AppendElement(&data);
        std::cout << "added element: " << i << std::endl;
    }
    // reading
    for (size_t i = 0; i < index; ++i) {
        MASSDataType read_out = xtra.ReadElement(i);
        std::cout << read_out.agent_id << ": " << static_cast<uint32_t>(read_out.front_rgb[0]) << std::endl;
    }
    auto [dset_size, dset_max_size] = xtra.GetCurrentSize();
    std::cout << dset_size << "/" << dset_max_size << std::endl;
    
    return 0;
}