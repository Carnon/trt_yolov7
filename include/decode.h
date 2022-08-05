#ifndef YOLOV5S_DECODE_H
#define YOLOV5S_DECODE_H

#include <cuda_runtime.h>
#include <cstdint>

void decode_output(float* src, float* dst, int input_width, int input_height, int num_class, float scale, cudaStream_t stream);

#endif //YOLOV5S_DECODE_H
