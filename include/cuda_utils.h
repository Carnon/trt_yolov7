//
// Created by carnon on 2022/6/21.
//

#ifndef YOLOV7_CUDA_UTILS_H
#define YOLOV7_CUDA_UTILS_H

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif

#endif //YOLOV7_CUDA_UTILS_H
