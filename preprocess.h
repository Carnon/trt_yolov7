//
// Created by carnon on 2022/6/20.
//

#ifndef HELLOWARPAFFINE_PREPROCESS_H
#define HELLOWARPAFFINE_PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>

void resize_img(uint8_t* src, int src_width, int src_height,
                float* dst, int dst_width, int dst_height,
                float scale, cudaStream_t stream);

#endif //HELLOWARPAFFINE_PREPROCESS_H
