#include <algorithm>
#include "cuda_utils.h"
#include "preprocess.h"
#include "decode.h"
#include "yolov7.h"

static Logger gLogger;

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASS = 80;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static const int BOX_SIZE = 80*80*3 + 40*40*3 + 20*20*3;

ICudaEngine* build_engine(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path){
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    // backbone
    IActivationLayer* relu0 = convBnRelu(network, weightMap, *data, 32, 3, 2, 1, "model.0");
    IActivationLayer* relu1 = convBnRelu(network, weightMap, *relu0->getOutput(0), 64, 3, 2, 1, "model.1");
    IActivationLayer* relu2 = convBnRelu(network, weightMap, *relu1->getOutput(0), 32, 1, 1, 0, "model.2");
    IActivationLayer* relu3 = convBnRelu(network, weightMap, *relu1->getOutput(0), 32, 1, 1, 0, "model.3");
    IActivationLayer* relu4 = convBnRelu(network, weightMap, *relu3->getOutput(0), 32, 3, 1, 1, "model.4");
    IActivationLayer* relu5 = convBnRelu(network, weightMap, *relu4->getOutput(0), 32, 3, 1, 1, "model.5");
    ITensor* input_tensor_6[] = {relu5->getOutput(0), relu4->getOutput(0), relu3->getOutput(0), relu2->getOutput(0)};
    IConcatenationLayer* cat6 = network->addConcatenation(input_tensor_6, 4);
    cat6->setAxis(0);
    IActivationLayer* relu7 = convBnRelu(network, weightMap, *cat6->getOutput(0), 64, 1, 1, 0, "model.7");

    IPoolingLayer* pool8 = network->addPoolingNd(*relu7->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool8->setStrideNd(DimsHW{2, 2});
    IActivationLayer* relu9 = convBnRelu(network, weightMap, *pool8->getOutput(0), 64, 1, 1, 0, "model.9");
    IActivationLayer* relu10 = convBnRelu(network, weightMap, *pool8->getOutput(0), 64, 1, 1, 0, "model.10");
    IActivationLayer* relu11 = convBnRelu(network, weightMap, *relu10->getOutput(0), 64, 3, 1, 1, "model.11");
    IActivationLayer* relu12 = convBnRelu(network, weightMap, *relu11->getOutput(0), 64, 3, 1, 1, "model.12");
    ITensor* input_tensor_13[] = {relu12->getOutput(0), relu11->getOutput(0), relu10->getOutput(0), relu9->getOutput(0)};
    IConcatenationLayer* cat13 = network->addConcatenation(input_tensor_13, 4);
    IActivationLayer* relu14 = convBnRelu(network, weightMap, *cat13->getOutput(0), 128, 1, 1, 0, "model.14");

    IPoolingLayer* pool15 = network->addPoolingNd(*relu14->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool15->setStrideNd(DimsHW{2, 2});
    IActivationLayer* relu16 = convBnRelu(network, weightMap, *pool15->getOutput(0), 128, 1, 1, 0, "model.16");
    IActivationLayer* relu17 = convBnRelu(network, weightMap, *pool15->getOutput(0), 128, 1, 1, 0, "model.17");
    IActivationLayer* relu18 = convBnRelu(network, weightMap, *relu17->getOutput(0), 128, 3, 1, 1, "model.18");
    IActivationLayer* relu19 = convBnRelu(network, weightMap, *relu18->getOutput(0), 128, 3, 1, 1, "model.19");
    ITensor* input_tensor_20[] = {relu19->getOutput(0), relu18->getOutput(0), relu17->getOutput(0), relu16->getOutput(0)};
    IConcatenationLayer* cat20 = network->addConcatenation(input_tensor_20, 4);
    cat20->setAxis(0);
    IActivationLayer* relu21 = convBnRelu(network, weightMap, *cat20->getOutput(0), 256, 1, 1, 0, "model.21");

    IPoolingLayer* pool22 = network->addPoolingNd(*relu21->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool22->setStrideNd(DimsHW{2, 2});
    IActivationLayer* relu23 = convBnRelu(network, weightMap, *pool22->getOutput(0), 256, 1, 1, 0, "model.23");
    IActivationLayer* relu24 = convBnRelu(network, weightMap, *pool22->getOutput(0), 256, 1, 1, 0, "model.24");
    IActivationLayer* relu25 = convBnRelu(network, weightMap, *relu24->getOutput(0), 256, 3, 1, 1, "model.25");
    IActivationLayer* relu26 = convBnRelu(network, weightMap, *relu25->getOutput(0), 256, 3, 1, 1, "model.26");
    ITensor* input_tensor_27[] = {relu26->getOutput(0), relu25->getOutput(0), relu24->getOutput(0), relu23->getOutput(0)};
    IConcatenationLayer* cat27 = network->addConcatenation(input_tensor_27, 4);
    cat27->setAxis(0);
    IActivationLayer* relu28 = convBnRelu(network, weightMap, *cat27->getOutput(0), 512, 1, 1, 0, "model.28");

    // head
    IActivationLayer* relu29 = convBnRelu(network, weightMap, *relu28->getOutput(0), 256, 1, 1, 0, "model.29");
    IActivationLayer* relu30 = convBnRelu(network, weightMap, *relu28->getOutput(0), 256, 1, 1, 0, "model.30");
    IPoolingLayer* pool31 = network->addPoolingNd(*relu30->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
    pool31->setPaddingNd(DimsHW{2, 2});
    pool31->setStrideNd(DimsHW{1, 1});
    IPoolingLayer* pool32 = network->addPoolingNd(*relu30->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
    pool32->setPaddingNd(DimsHW{4, 4});
    pool32->setStrideNd(DimsHW{1, 1});
    IPoolingLayer* pool33 = network->addPoolingNd(*relu30->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
    pool33->setPaddingNd(DimsHW{6, 6});
    pool33->setStrideNd(DimsHW{1, 1});
    ITensor* input_tensor_34[] = {pool33->getOutput(0), pool32->getOutput(0), pool31->getOutput(0), relu30->getOutput(0)};
    IConcatenationLayer* cat34 = network->addConcatenation(input_tensor_34, 4);
    cat34->setAxis(0);
    IActivationLayer* relu35 = convBnRelu(network, weightMap, *cat34->getOutput(0), 256, 1, 1, 0, "model.35");
    ITensor* input_tensor_36[] = {relu35->getOutput(0), relu29->getOutput(0)};
    IConcatenationLayer* cat36 = network->addConcatenation(input_tensor_36, 2);
    cat36->setAxis(0);
    IActivationLayer* relu37 = convBnRelu(network, weightMap, *cat36->getOutput(0), 256, 1, 1, 0, "model.37");

    IActivationLayer* relu38 = convBnRelu(network, weightMap, *relu37->getOutput(0), 128 , 1, 1, 0, "model.38");
    float scale[] = {1.0, 2.0, 2.0};
    IResizeLayer* resize39 = network->addResize(*relu38->getOutput(0));
    resize39->setResizeMode(ResizeMode::kNEAREST);
    resize39->setScales(scale, 3);
    IActivationLayer* relu40 = convBnRelu(network, weightMap, *relu21->getOutput(0), 128, 1, 1, 0, "model.40");
    ITensor* input_tensor_41[] = {relu40->getOutput(0), resize39->getOutput(0)};
    IConcatenationLayer* cat41 = network->addConcatenation(input_tensor_41, 2);
    cat41->setAxis(0);

    IActivationLayer* relu42 = convBnRelu(network, weightMap, *cat41->getOutput(0), 64, 1, 1, 0, "model.42");
    IActivationLayer* relu43 = convBnRelu(network, weightMap, *cat41->getOutput(0), 64, 1, 1, 0, "model.43");
    IActivationLayer* relu44 = convBnRelu(network, weightMap, *relu43->getOutput(0), 64, 3, 1, 1, "model.44");
    IActivationLayer* relu45 = convBnRelu(network, weightMap, *relu44->getOutput(0), 64, 3, 1, 1, "model.45");
    ITensor* input_tensor_46[] = {relu45->getOutput(0), relu44->getOutput(0), relu43->getOutput(0), relu42->getOutput(0)};
    IConcatenationLayer* cat46 = network->addConcatenation(input_tensor_46, 4);
    cat46->setAxis(0);
    IActivationLayer* relu47 = convBnRelu(network, weightMap, *cat46->getOutput(0), 128, 1, 1, 0, "model.47");

    IActivationLayer* relu48 = convBnRelu(network, weightMap, *relu47->getOutput(0), 64, 1, 1, 0, "model.48");
    IResizeLayer* resize49 = network->addResize(*relu48->getOutput(0));
    resize49->setResizeMode(ResizeMode::kNEAREST);
    resize49->setScales(scale, 3);
    IActivationLayer* relu50 = convBnRelu(network, weightMap, *relu14->getOutput(0), 64, 1, 1, 0, "model.50");
    ITensor* input_tensor_51[] = {relu50->getOutput(0), resize49->getOutput(0)};
    IConcatenationLayer* cat51 = network->addConcatenation(input_tensor_51, 2);
    cat51->setAxis(0);

    IActivationLayer* relu52 = convBnRelu(network, weightMap, *cat51->getOutput(0), 32, 1, 1, 0, "model.52");
    IActivationLayer* relu53 = convBnRelu(network, weightMap, *cat51->getOutput(0), 32, 1, 1, 0, "model.53");
    IActivationLayer* relu54 = convBnRelu(network, weightMap, *relu53->getOutput(0), 32, 3, 1, 1, "model.54");
    IActivationLayer* relu55 = convBnRelu(network, weightMap, *relu54->getOutput(0), 32, 3, 1, 1, "model.55");
    ITensor* input_tensor_56[] = {relu55->getOutput(0), relu54->getOutput(0), relu53->getOutput(0), relu52->getOutput(0)};
    IConcatenationLayer* cat56 = network->addConcatenation(input_tensor_56, 4);
    cat56->setAxis(0);
    IActivationLayer* relu57 = convBnRelu(network, weightMap, *cat56->getOutput(0), 64, 1, 1, 0, "model.57");

    IActivationLayer* relu58 = convBnRelu(network, weightMap, *relu57->getOutput(0), 128 , 3, 2, 1, "model.58");
    ITensor* input_tensor_59[] = {relu58->getOutput(0), relu47->getOutput(0)};
    IConcatenationLayer* cat59 = network->addConcatenation(input_tensor_59, 2);
    cat59->setAxis(0);

    IActivationLayer* relu60 = convBnRelu(network, weightMap, *cat59->getOutput(0), 64, 1, 1, 0, "model.60");
    IActivationLayer* relu61 = convBnRelu(network, weightMap, *cat59->getOutput(0), 64, 1, 1, 0, "model.61");
    IActivationLayer* relu62 = convBnRelu(network, weightMap, *relu61->getOutput(0), 64, 3, 1, 1, "model.62");
    IActivationLayer* relu63 = convBnRelu(network, weightMap, *relu62->getOutput(0), 64, 3, 1, 1, "model.63");
    ITensor* input_tensor_64[] = {relu63->getOutput(0), relu62->getOutput(0), relu61->getOutput(0), relu60->getOutput(0)};
    IConcatenationLayer* cat64 = network->addConcatenation(input_tensor_64, 4);
    cat64->setAxis(0);
    IActivationLayer* relu65 = convBnRelu(network, weightMap, *cat64->getOutput(0), 128, 1, 1, 0, "model.65");

    IActivationLayer* relu66 = convBnRelu(network, weightMap, *relu65->getOutput(0), 256, 3, 2, 1, "model.66");
    ITensor* input_tensor_67[] = {relu66->getOutput(0), relu37->getOutput(0)};
    IConcatenationLayer* cat67 = network->addConcatenation(input_tensor_67, 2);
    cat67->setAxis(0);

    IActivationLayer* relu68 = convBnRelu(network, weightMap, *cat67->getOutput(0), 128, 1, 1, 0, "model.68");
    IActivationLayer* relu69 = convBnRelu(network, weightMap, *cat67->getOutput(0), 128, 1, 1, 0, "model.69");
    IActivationLayer* relu70 = convBnRelu(network, weightMap, *relu69->getOutput(0), 128, 3, 1, 1, "model.70");
    IActivationLayer* relu71 = convBnRelu(network, weightMap, *relu70->getOutput(0), 128, 3, 1, 1, "model.71");
    ITensor* input_tensor_72[] = {relu71->getOutput(0), relu70->getOutput(0), relu69->getOutput(0), relu68->getOutput(0)};
    IConcatenationLayer* cat72 = network->addConcatenation(input_tensor_72, 4);
    cat72->setAxis(0);
    IActivationLayer* relu73 = convBnRelu(network, weightMap, *cat72->getOutput(0), 256, 1, 1, 0, "model.73");

    IActivationLayer* relu74 = convBnRelu(network, weightMap, *relu57->getOutput(0), 128, 3, 1, 1, "model.74");
    IActivationLayer* relu75 = convBnRelu(network, weightMap, *relu65->getOutput(0), 256, 3, 1, 1, "model.75");
    IActivationLayer* relu76 = convBnRelu(network, weightMap, *relu73->getOutput(0), 512, 3, 1, 1, "model.76");

    // out
    IConvolutionLayer* cv77_0 = network->addConvolutionNd(*relu74->getOutput(0), 3*(NUM_CLASS+5), DimsHW{1, 1}, weightMap["model.77.m.0.weight"], weightMap["model.77.m.0.bias"]);
    assert(cv77_0);
    cv77_0->setName("cv77.0");
    IConvolutionLayer* cv77_1 = network->addConvolutionNd(*relu75->getOutput(0), 3*(NUM_CLASS+5), DimsHW{1, 1}, weightMap["model.77.m.1.weight"], weightMap["model.77.m.1.bias"]);
    assert(cv77_1);
    cv77_1->setName("cv77.1");
    IConvolutionLayer* cv77_2 = network->addConvolutionNd(*relu76->getOutput(0), 3*(NUM_CLASS+5), DimsHW{1, 1}, weightMap["model.77.m.2.weight"], weightMap["model.77.m.2.bias"]);
    assert(cv77_2);
    cv77_2->setName("cv77.2");

    IShuffleLayer* sf0 = network->addShuffle(*cv77_0->getOutput(0));
    sf0->setReshapeDimensions(Dims4{3, NUM_CLASS+5, INPUT_H/8, INPUT_W/8});
    sf0->setSecondTranspose(Permutation{0, 2, 3, 1});
    IActivationLayer* act0 = network->addActivation(*sf0->getOutput(0), ActivationType::kSIGMOID);
    IShuffleLayer* out0 = network->addShuffle(*act0->getOutput(0));
    out0->setReshapeDimensions(DimsHW{3*INPUT_H/8*INPUT_W/8, NUM_CLASS+5});

    IShuffleLayer* sf1 = network->addShuffle(*cv77_1->getOutput(0));
    sf1->setReshapeDimensions(Dims4{3, NUM_CLASS+5, INPUT_H/16, INPUT_W/16});
    sf1->setSecondTranspose(Permutation{0, 2, 3, 1});
    IActivationLayer* act1 = network->addActivation(*sf1->getOutput(0), ActivationType::kSIGMOID);
    IShuffleLayer* out1 = network->addShuffle(*act1->getOutput(0));
    out1->setReshapeDimensions(DimsHW{3*INPUT_H/16*INPUT_W/16, NUM_CLASS+5});

    IShuffleLayer* sf2 = network->addShuffle(*cv77_2->getOutput(0));
    sf2->setReshapeDimensions(Dims4{3, NUM_CLASS+5, INPUT_H/32, INPUT_W/32});
    sf2->setSecondTranspose(Permutation{0, 2, 3, 1});
    IActivationLayer* act2 = network->addActivation(*sf2->getOutput(0), ActivationType::kSIGMOID);
    IShuffleLayer* out2 = network->addShuffle(*act2->getOutput(0));
    out2->setReshapeDimensions(DimsHW{3*INPUT_H/32*INPUT_W/32, NUM_CLASS+5});

    ITensor* outputTensors[] = {out0->getOutput(0), out1->getOutput(0), out2->getOutput(0)};
    IConcatenationLayer* output = network->addConcatenation(outputTensors, 3);
    output->setAxis(0);

    output->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*output->getOutput(0));

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize((1<<30));
    config->setFlag(BuilderFlag::kFP16);

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return engine;
}

void serialize(const std::string& wts_path, const std::string& engine_path){
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = build_engine(builder, config, DataType::kFLOAT, wts_path);
    assert(engine != nullptr);
    IHostMemory* modelStream = engine->serialize();
    assert(modelStream != nullptr);
    std::ofstream p(engine_path, std::ios::binary);
    p.write(static_cast<const char *>(modelStream->data()), modelStream->size());

    modelStream->destroy();
    engine->destroy();
    builder->destroy();
    std::cout<<"convert ok!"<<std::endl;
}

void hello() {
    std::cout << "Hello, World!" << std::endl;
}

void loadEngine(const char* engine_path){
    std::string cached_engine;
    std::fstream file;
    file.open(engine_path, std::ios::in);
    if(!file.is_open()){
        std::cout<<"read engine file error"<<std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF){
        std::stringstream  buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    bufferSize[0] = 3*INPUT_H*INPUT_W*sizeof(float);
    bufferSize[1] = BOX_SIZE*(NUM_CLASS+5)*sizeof(float);
    int max_image_size = 3000*3000*3*sizeof(float);

    CUDA_CHECK(cudaMalloc(&psrc_device, max_image_size));
    CUDA_CHECK(cudaMalloc(&buffers[0], bufferSize[0]));
    CUDA_CHECK(cudaMalloc(&buffers[1], bufferSize[1]));
    CUDA_CHECK(cudaMalloc(&pdst_device, BOX_SIZE*6*sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::cout<<"load engine ok! "<<std::endl;
}

void release(){
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(psrc_device));
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    CUDA_CHECK(cudaFree(pdst_device));
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

int inferImage(uint8_t* data, int w, int h, float* result){
    float scale = std::min(float(INPUT_H)/ float(h), float(INPUT_W) / float(w));
    CUDA_CHECK(cudaMemcpy(psrc_device, data, w*h*3, cudaMemcpyHostToDevice));

    // 1. prepare img  仿射变换对图片resize大小 基于cuda实现
    resize_img(psrc_device, w, h, buffers[0], INPUT_W, INPUT_H, scale, stream);

    // 2. model infer 模型推理
    auto *output = new float[BOX_SIZE*6];
    context->enqueue(1, (void **)buffers, stream, nullptr);
    // 3. 解析yolo输出，使用cuda代码实现。
    decode_output(buffers[1], pdst_device, INPUT_H, INPUT_W, 1/scale, stream);
    cudaMemcpyAsync(output, pdst_device, BOX_SIZE*6*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    //4. 后处理
    auto boxes = postProcess(output);

    int size = int(boxes.size());
    if(size > 1000) size = 1000;
    for(int i=0; i<size; i++){
        result[0+6*i] = boxes[i].x;
        result[1+6*i] = boxes[i].y;
        result[2+6*i] = boxes[i].w;
        result[3+6*i] = boxes[i].h;
        result[4+6*i] = float(boxes[i].classId);
        result[5+6*i] = boxes[i].conf;
    }

    delete[] output;
    return size;
}

float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b) {
    float xx1 = std::max(det_a.x - det_a.w/2, det_b.x - det_b.w/2);
    float yy1 = std::max(det_a.y - det_a.h/2, det_b.y - det_b.h/2);
    float xx2 = std::min(det_a.x + det_a.w/2, det_b.x + det_b.w/2);
    float yy2 = std::min(det_a.y + det_a.h/2, det_b.y + det_b.h/2);

    if(xx1 > xx2 || yy1 > yy2) return 0.0;

    float inter = (xx2 - xx1 + 1) * (yy2 - yy1 + 1);
    return inter / (det_a.w * det_a.h + det_b.w * det_b.h - inter);
}

void NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.conf > right.conf;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
            if (detections[i].classId == detections[j].classId){
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > 0.35) detections[j].conf = 0.0;
            }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det){
        return det.conf == 0.0; }), detections.end());
}

std::vector<DetectRes> postProcess(float *output){
    std::vector<DetectRes> result;
    for(int i=0; i<BOX_SIZE; i++){
        float* row = output + 6*i;
        if(row[5] > 0.5){
            DetectRes res;
            res.x = row[0];
            res.y = row[1];
            res.w = row[2];
            res.h = row[3];
            res.classId = int(row[4]);
            res.conf = row[5];
            result.push_back(res);
        }
    }
    NmsDetect(result);
    return result;
}
