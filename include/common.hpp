#ifndef YOLOV7_COMMON_H
#define YOLOV7_COMMON_H

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "NvInfer.h"
#include <cassert>
#include <cmath>

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val;
        val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// batch norm
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    scale_1->setName(lname.c_str());
    return scale_1;
}

// Conv
IActivationLayer* convBnRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, int p, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname+".conv").c_str());
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname+".bn", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu1);
    relu1->setAlpha(0.1);
    return relu1;
}

#endif //YOLOV7_COMMON_H
