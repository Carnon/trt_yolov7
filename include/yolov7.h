#ifndef TRT_YOLOV7_H
#define TRT_YOLOV7_H

#include <iostream>
#include "common.hpp"
#include "logging.h"
#include "decodelayer.h"


ICudaEngine* build_engine(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path);

void serialize(const std::string& wts_path, const std::string& engine_path);

typedef struct DetectRes{
    float x, y, w, h;
    int classId;
    float conf;
} DetectRes;

nvinfer1::IRuntime* runtime = nullptr;
nvinfer1::ICudaEngine* engine = nullptr;
nvinfer1::IExecutionContext* context = nullptr;
cudaStream_t stream;
uint8_t* psrc_device = nullptr;
int64_t bufferSize[2];
float* pdst_device = nullptr;
float* buffers[2];

#ifdef __cplusplus
extern "C" {
#endif

void loadEngine(const char *engine_path);
int inferImage(uint8_t *image, int w, int h, float *result);
void release();

#ifdef __cplusplus
};
#endif

float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
void NmsDetect(std::vector<DetectRes> &detections);
std::vector<DetectRes> postProcess(float *output, float scale);

void hello();
#endif //TRT_YOLOV7_H
