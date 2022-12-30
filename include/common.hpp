#ifndef YOLOV7_COMMON_H
#define YOLOV7_COMMON_H

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include "decodelayer.h"

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


std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname){
    Weights wts = weightMap[lname+".anchor_grid"];
    auto *p = (const float*)wts.values;
    std::vector<float> anchors(p, p+18);
    return anchors;
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
IElementWiseLayer* convBnSilu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, int p, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname+".conv").c_str());
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname+".bn", 1e-5);

    // silu = x * sigmoid(x)
    IActivationLayer* sig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig1);
    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *sig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
}

// SPPCSPC
IElementWiseLayer* SPPCSPC(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, const std::string& lname){
    int c_ = int(2 * c2 * 0.5);
    IElementWiseLayer* cv1 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, lname+".cv1");
    IElementWiseLayer* cv2 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, lname+".cv2");

    IElementWiseLayer* cv3 = convBnSilu(network, weightMap, *cv1->getOutput(0), c_, 3, 1, 1, lname+".cv3");
    IElementWiseLayer* cv4 = convBnSilu(network, weightMap, *cv3->getOutput(0), c_, 1, 1, 0, lname+".cv4");

    IPoolingLayer* m1 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
    m1->setStrideNd(DimsHW{1, 1});
    m1->setPaddingNd(DimsHW{2, 2});
    IPoolingLayer* m2 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
    m2->setStrideNd(DimsHW{1, 1});
    m2->setPaddingNd(DimsHW{4, 4});
    IPoolingLayer* m3 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
    m3->setStrideNd(DimsHW{1, 1});
    m3->setPaddingNd(DimsHW{6, 6});

    ITensor* input_tensors[] = {cv4->getOutput(0), m1->getOutput(0), m2->getOutput(0), m3->getOutput(0)};
    IConcatenationLayer* concat = network->addConcatenation(input_tensors, 4);
    // 0U
    concat->setAxis(0);

    IElementWiseLayer* cv5 = convBnSilu(network, weightMap, *concat->getOutput(0), c_, 1, 1, 0, lname+".cv5");
    IElementWiseLayer* cv6 = convBnSilu(network, weightMap, *cv5->getOutput(0), c_, 3, 1, 1, lname+".cv6");

    ITensor* input_tensors2[] = {cv6->getOutput(0), cv2->getOutput(0)};
    IConcatenationLayer* concat1 = network->addConcatenation(input_tensors2, 2);
    // 0U
    concat1->setAxis(0);


    IElementWiseLayer* cv7 = convBnSilu(network, weightMap, *concat1->getOutput(0), c2, 1, 1, 0, lname+".cv7");
    return cv7;
}

// RepConv
IElementWiseLayer* RepConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, const std::string& lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // 256 * 128 * 3 *3
    IConvolutionLayer* rbr_dense_conv = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname+".rbr_dense.0.weight"], emptywts);
    assert(rbr_dense_conv);
    rbr_dense_conv->setPaddingNd(DimsHW{k/2, k/2});
    rbr_dense_conv->setStrideNd(DimsHW{s, s});
    rbr_dense_conv->setName((lname+".rbr_dense.0").c_str());
    IScaleLayer* rbr_dense_bn = addBatchNorm2d(network, weightMap, *rbr_dense_conv->getOutput(0), lname+".rbr_dense.1", 1e-3);

    IConvolutionLayer* rbr_1x1_conv = network->addConvolutionNd(input, c2, DimsHW{1, 1}, weightMap[lname+".rbr_1x1.0.weight"], emptywts);
    assert(rbr_1x1_conv);
    rbr_1x1_conv->setStrideNd(DimsHW{s, s});
    rbr_1x1_conv->setName((lname+".rbr_1x1.0").c_str());
    IScaleLayer* rbr_1x1_bn = addBatchNorm2d(network, weightMap, *rbr_1x1_conv->getOutput(0), lname+".rbr_1x1.1", 1e-3);

    IElementWiseLayer* ew1 = network->addElementWise(*rbr_dense_bn->getOutput(0), *rbr_1x1_bn->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    // silu
    IActivationLayer* sigmoid = network->addActivation(*ew1->getOutput(0), ActivationType::kSIGMOID);
    IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    return ew2;
}

// DecodeLayer
IPluginV2Layer* addDecodeLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, const std::vector<IActivationLayer*>& dets, int nbInputs, int class_num, int input_w, int input_h, int max_out_object, std::string lname){

    auto creator = getPluginRegistry()->getPluginCreator("DecodeLayer_TRT", "1");
    PluginField pluginFields[2];
    int netInfo[4] = {class_num, input_w, input_h, max_out_object};
    pluginFields[0].data = netInfo;
    pluginFields[0].length = 4;
    pluginFields[0].name = "netInfo";
    pluginFields[0].type = PluginFieldType::kFLOAT32;

    std::vector<float> anchors = getAnchors(weightMap, lname);
    pluginFields[1].data = &anchors[0];
    pluginFields[1].length = (int)anchors.size();
    pluginFields[1].name = "anchor";
    pluginFields[1].type = PluginFieldType::kFLOAT32;

    PluginFieldCollection plugin_data{};
    plugin_data.nbFields = 2;
    plugin_data.fields = pluginFields;

    IPluginV2 *plugin_obj = creator->createPlugin("decodeLayer", &plugin_data);

    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }

    auto output = network->addPluginV2(&input_tensors[0], nbInputs, *plugin_obj);
    return output;
}

#endif //YOLOV7_COMMON_H
