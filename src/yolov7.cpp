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
    IElementWiseLayer* ew0 = convBnSilu(network, weightMap, *data, 32, 3, 1, 1, "model.0");

    IElementWiseLayer* ew1 = convBnSilu(network, weightMap, *ew0->getOutput(0), 64, 3, 2, 1, "model.1");
    IElementWiseLayer* ew2 = convBnSilu(network, weightMap, *ew1->getOutput(0), 64, 3, 1, 1, "model.2");

    IElementWiseLayer* ew3 = convBnSilu(network, weightMap, *ew2->getOutput(0), 128, 3, 2, 1, "model.3");
    IElementWiseLayer* ew4 = convBnSilu(network, weightMap, *ew3->getOutput(0), 64, 1, 1, 0, "model.4");
    IElementWiseLayer* ew5 = convBnSilu(network, weightMap, *ew3->getOutput(0), 64, 1, 1, 0, "model.5");
    IElementWiseLayer* ew6 = convBnSilu(network, weightMap, *ew5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* ew7 = convBnSilu(network, weightMap, *ew6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* ew8 = convBnSilu(network, weightMap, *ew7->getOutput(0), 64, 3, 1, 1, "model.8");
    IElementWiseLayer* ew9 = convBnSilu(network, weightMap, *ew8->getOutput(0), 64, 3, 1, 1, "model.9");
    ITensor* input_tensor_10[] = {ew9->getOutput(0), ew7->getOutput(0), ew5->getOutput(0), ew4->getOutput(0)};
    IConcatenationLayer* concat10 = network->addConcatenation(input_tensor_10, 4);
    concat10->setAxis(0);
    IElementWiseLayer* ew11 = convBnSilu(network, weightMap, *concat10->getOutput(0), 256, 1, 1, 0, "model.11");

    IPoolingLayer* mp12 = network->addPoolingNd(*ew11->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    mp12->setStrideNd(DimsHW{2, 2});
    IElementWiseLayer* ew13 = convBnSilu(network, weightMap, *mp12->getOutput(0), 128, 1, 1, 0, "model.13");
    IElementWiseLayer* ew14 = convBnSilu(network, weightMap, *ew11->getOutput(0), 128, 1, 1, 0, "model.14");
    IElementWiseLayer* ew15 = convBnSilu(network, weightMap, *ew14->getOutput(0), 128, 3, 2, 1, "model.15");
    ITensor* input_tensor_16[] = {ew15->getOutput(0), ew13->getOutput(0)};
    IConcatenationLayer* concat16 = network->addConcatenation(input_tensor_16, 2);
    IElementWiseLayer* ew17 = convBnSilu(network, weightMap, *concat16->getOutput(0), 128, 1, 1, 0, "model.17");
    IElementWiseLayer* ew18 = convBnSilu(network, weightMap, *concat16->getOutput(0), 128 ,1, 1, 0, "model.18");
    IElementWiseLayer* ew19 = convBnSilu(network, weightMap, *ew18->getOutput(0), 128, 3, 1, 1, "model.19");
    IElementWiseLayer* ew20 = convBnSilu(network, weightMap, *ew19->getOutput(0), 128, 3, 1, 1, "model.20");
    IElementWiseLayer* ew21 = convBnSilu(network, weightMap, *ew20->getOutput(0), 128, 3, 1, 1, "model.21");
    IElementWiseLayer* ew22 = convBnSilu(network, weightMap, *ew21->getOutput(0), 128, 3, 1, 1, "model.22");
    ITensor* input_tensor_23[] = {ew22->getOutput(0), ew20->getOutput(0), ew18->getOutput(0), ew17->getOutput(0)};
    IConcatenationLayer* concat23 = network->addConcatenation(input_tensor_23, 4);
    concat23->setAxis(0);
    IElementWiseLayer* ew24 = convBnSilu(network, weightMap, *concat23->getOutput(0), 512, 1, 1, 0, "model.24");

    IPoolingLayer* mp25 = network->addPoolingNd(*ew24->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    mp25->setStrideNd(DimsHW{2, 2});
    IElementWiseLayer* ew26 = convBnSilu(network, weightMap, *mp25->getOutput(0), 256, 1, 1, 0, "model.26");
    IElementWiseLayer* ew27 = convBnSilu(network, weightMap, *ew24->getOutput(0), 256, 1, 1, 0, "model.27");
    IElementWiseLayer* ew28 = convBnSilu(network, weightMap, *ew27->getOutput(0), 256, 3, 2, 1, "model.28");
    ITensor* input_tensor_29[] = {ew28->getOutput(0), ew26->getOutput(0)};
    IConcatenationLayer* concat29 = network->addConcatenation(input_tensor_29, 2);
    IElementWiseLayer* ew30 = convBnSilu(network, weightMap, *concat29->getOutput(0), 256, 1, 1, 0, "model.30");
    IElementWiseLayer* ew31 = convBnSilu(network, weightMap, *concat29->getOutput(0), 256, 1, 1, 0, "model.31");
    IElementWiseLayer* ew32 = convBnSilu(network, weightMap, *ew31->getOutput(0), 256, 3, 1, 1, "model.32");
    IElementWiseLayer* ew33 = convBnSilu(network, weightMap, *ew32->getOutput(0), 256, 3, 1, 1, "model.33");
    IElementWiseLayer* ew34 = convBnSilu(network, weightMap, *ew33->getOutput(0), 256, 3, 1, 1, "model.34");
    IElementWiseLayer* ew35 = convBnSilu(network, weightMap, *ew34->getOutput(0), 256, 3, 1, 1, "model.35");
    ITensor* input_tensor_36[] = {ew35->getOutput(0), ew33->getOutput(0), ew31->getOutput(0), ew30->getOutput(0)};
    IConcatenationLayer* concat36 = network->addConcatenation(input_tensor_36, 4);
    concat36->setAxis(0);
    IElementWiseLayer* ew37 = convBnSilu(network, weightMap, *concat36->getOutput(0), 1024, 1, 1, 0, "model.37");

    IPoolingLayer* mp38 = network->addPoolingNd(*ew37->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    mp38->setStrideNd(DimsHW{2, 2});
    IElementWiseLayer* ew39 = convBnSilu(network, weightMap, *mp38->getOutput(0), 512, 1, 1, 0, "model.39");
    IElementWiseLayer* ew40 = convBnSilu(network, weightMap, *ew37->getOutput(0), 512, 1, 1, 0, "model.40");
    IElementWiseLayer* ew41 = convBnSilu(network, weightMap, *ew40->getOutput(0), 512, 3, 2, 1, "model.41");
    ITensor* input_tensor_42[] = {ew41->getOutput(0), ew39->getOutput(0)};
    IConcatenationLayer* concat42 = network->addConcatenation(input_tensor_42, 2);
    concat42->setAxis(0);
    IElementWiseLayer* ew43 = convBnSilu(network, weightMap, *concat42->getOutput(0), 256, 1, 1, 0, "model.43");
    IElementWiseLayer* ew44 = convBnSilu(network, weightMap, *concat42->getOutput(0), 256, 1, 1, 0, "model.44");
    IElementWiseLayer* ew45 = convBnSilu(network, weightMap, *ew44->getOutput(0), 256, 3, 1, 1, "model.45");
    IElementWiseLayer* ew46 = convBnSilu(network, weightMap, *ew45->getOutput(0), 256, 3, 1, 1, "model.46");
    IElementWiseLayer* ew47 = convBnSilu(network, weightMap, *ew46->getOutput(0), 256, 3, 1, 1, "model.47");
    IElementWiseLayer* ew48 = convBnSilu(network, weightMap, *ew47->getOutput(0), 256, 3, 1, 1, "model.48");
    ITensor* input_tensor_49[] = {ew48->getOutput(0), ew46->getOutput(0), ew44->getOutput(0), ew43->getOutput(0)};
    IConcatenationLayer* concat49 = network->addConcatenation(input_tensor_49, 4);
    concat49->setAxis(0);
    IElementWiseLayer* ew50 = convBnSilu(network, weightMap, *concat49->getOutput(0), 1024, 1, 1, 0, "model.50");

    // head
    IElementWiseLayer* ew51 = SPPCSPC(network, weightMap, *ew50->getOutput(0), 512, "model.51");

    IElementWiseLayer* ew52 = convBnSilu(network, weightMap, *ew51->getOutput(0), 256, 1, 1, 0, "model.52");
    float scale[] = {1.0, 2.0, 2.0};
    IResizeLayer* re53 = network->addResize(*ew52->getOutput(0));
    re53->setResizeMode(ResizeMode::kNEAREST);
    re53->setScales(scale, 3);
    IElementWiseLayer* ew54 = convBnSilu(network, weightMap, *ew37->getOutput(0), 256, 1, 1, 0, "model.54");
    ITensor* input_tensor_55[] = {ew54->getOutput(0), re53->getOutput(0)};
    IConcatenationLayer* concat55 = network->addConcatenation(input_tensor_55, 2);
    concat55->setAxis(0);

    IElementWiseLayer* ew56 = convBnSilu(network, weightMap, *concat55->getOutput(0), 256, 1, 1, 0, "model.56");
    IElementWiseLayer* ew57 = convBnSilu(network, weightMap, *concat55->getOutput(0), 256, 1, 1, 0, "model.57");
    IElementWiseLayer* ew58 = convBnSilu(network, weightMap, *ew57->getOutput(0), 128, 3, 1, 1, "model.58");
    IElementWiseLayer* ew59 = convBnSilu(network, weightMap, *ew58->getOutput(0), 128, 3, 1, 1, "model.59");
    IElementWiseLayer* ew60 = convBnSilu(network, weightMap, *ew59->getOutput(0), 128, 3, 1, 1, "model.60");
    IElementWiseLayer* ew61 = convBnSilu(network, weightMap, *ew60->getOutput(0), 128, 3, 1, 1, "model.61");
    ITensor* input_tensor_62[] = {ew61->getOutput(0), ew60->getOutput(0), ew59->getOutput(0), ew58->getOutput(0), ew57->getOutput(0), ew56->getOutput(0)};
    IConcatenationLayer* concat62 = network->addConcatenation(input_tensor_62, 6);
    IElementWiseLayer* ew63 = convBnSilu(network, weightMap, *concat62->getOutput(0), 256, 1, 1, 0, "model.63");

    IElementWiseLayer* ew64 = convBnSilu(network, weightMap, *ew63->getOutput(0), 128, 1, 1, 0, "model.64");
    IResizeLayer* re65 = network->addResize(*ew64->getOutput(0));
    re65->setResizeMode(ResizeMode::kNEAREST);
    re65->setScales(scale, 3);
    IElementWiseLayer* ew66 = convBnSilu(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, "model.66");
    ITensor* input_tensor_67[] = {ew66->getOutput(0), re65->getOutput(0)};
    IConcatenationLayer* concat67 = network->addConcatenation(input_tensor_67, 2);
    concat67->setAxis(0);

    IElementWiseLayer* ew68 = convBnSilu(network, weightMap, *concat67->getOutput(0), 128, 1, 1, 0, "model.68");
    IElementWiseLayer* ew69 = convBnSilu(network, weightMap, *concat67->getOutput(0), 128, 1, 1, 0, "model.69");
    IElementWiseLayer* ew70 = convBnSilu(network, weightMap, *ew69->getOutput(0), 64, 3, 1, 1, "model.70");
    IElementWiseLayer* ew71 = convBnSilu(network, weightMap, *ew70->getOutput(0), 64, 3, 1, 1, "model.71");
    IElementWiseLayer* ew72 = convBnSilu(network, weightMap, *ew71->getOutput(0), 64, 3, 1, 1, "model.72");
    IElementWiseLayer* ew73 = convBnSilu(network, weightMap, *ew72->getOutput(0), 64, 3, 1, 1, "model.73");
    ITensor* input_tensor_74[] = {ew73->getOutput(0), ew72->getOutput(0), ew71->getOutput(0), ew70->getOutput(0), ew69->getOutput(0), ew68->getOutput(0)};
    IConcatenationLayer* concat74 = network->addConcatenation(input_tensor_74, 6);
    concat74->setAxis(0);
    IElementWiseLayer* ew75 = convBnSilu(network, weightMap, *concat74->getOutput(0), 128, 1, 1, 0, "model.75");

    IPoolingLayer* mp76 = network->addPoolingNd(*ew75->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    mp76->setStrideNd(DimsHW{2, 2});
    IElementWiseLayer* ew77 = convBnSilu(network, weightMap, *mp76->getOutput(0), 128, 1, 1, 0, "model.77");
    IElementWiseLayer* ew78 = convBnSilu(network, weightMap, *ew75->getOutput(0), 128, 1, 1, 0, "model.78");
    IElementWiseLayer* ew79 = convBnSilu(network, weightMap, *ew78->getOutput(0), 128, 3, 2, 1, "model.79");
    ITensor* input_tensor_80[] = {ew79->getOutput(0), ew77->getOutput(0), ew63->getOutput(0)};
    IConcatenationLayer* concat80 = network->addConcatenation(input_tensor_80, 3);
    concat80->setAxis(0);

    IElementWiseLayer* ew81 = convBnSilu(network, weightMap, *concat80->getOutput(0), 256, 1, 1, 0, "model.81");
    IElementWiseLayer* ew82 = convBnSilu(network, weightMap, *concat80->getOutput(0), 256, 1, 1, 0, "model.82");
    IElementWiseLayer* ew83 = convBnSilu(network, weightMap, *ew82->getOutput(0), 128, 3, 1, 1, "model.83");
    IElementWiseLayer* ew84 = convBnSilu(network, weightMap, *ew83->getOutput(0), 128, 3, 1, 1, "model.84");
    IElementWiseLayer* ew85 = convBnSilu(network, weightMap, *ew84->getOutput(0), 128 ,3, 1, 1, "model.85");
    IElementWiseLayer* ew86 = convBnSilu(network, weightMap, *ew85->getOutput(0), 128, 3, 1, 1, "model.86");
    ITensor* input_tensor_87[] = {ew86->getOutput(0), ew85->getOutput(0), ew84->getOutput(0), ew83->getOutput(0), ew82->getOutput(0), ew81->getOutput(0)};
    IConcatenationLayer* concat87 = network->addConcatenation(input_tensor_87, 6);
    concat87->setAxis(0);
    IElementWiseLayer* ew88 = convBnSilu(network, weightMap, *concat87->getOutput(0), 256, 1, 1, 0, "model.88");

    IPoolingLayer* mp89 = network->addPoolingNd(*ew88->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    mp89->setStrideNd(DimsHW{2, 2});
    IElementWiseLayer* ew90 = convBnSilu(network, weightMap, *mp89->getOutput(0), 256, 1, 1, 0, "model.90");
    IElementWiseLayer* ew91 = convBnSilu(network, weightMap, *ew88->getOutput(0), 256, 1, 1, 0, "model.91");
    IElementWiseLayer* ew92 = convBnSilu(network, weightMap, *ew91->getOutput(0), 256, 3, 2, 1, "model.92");
    ITensor* input_tensor_93[] = {ew92->getOutput(0), ew90->getOutput(0), ew51->getOutput(0)};
    IConcatenationLayer* concat93 = network->addConcatenation(input_tensor_93, 3);
    concat93->setAxis(0);

    IElementWiseLayer* ew94 = convBnSilu(network, weightMap, *concat93->getOutput(0), 512, 1, 1, 0, "model.94");
    IElementWiseLayer* ew95 = convBnSilu(network, weightMap, *concat93->getOutput(0), 512, 1, 1, 0, "model.95");
    IElementWiseLayer* ew96 = convBnSilu(network, weightMap, *ew95->getOutput(0), 256, 3, 1, 1, "model.96");
    IElementWiseLayer* ew97 = convBnSilu(network, weightMap, *ew96->getOutput(0), 256, 3, 1, 1, "model.97");
    IElementWiseLayer* ew98 = convBnSilu(network, weightMap, *ew97->getOutput(0), 256, 3, 1, 1, "model.98");
    IElementWiseLayer* ew99 = convBnSilu(network, weightMap, *ew98->getOutput(0), 256, 3, 1, 1, "model.99");
    ITensor* input_tensor_100[] = {ew99->getOutput(0), ew98->getOutput(0), ew97->getOutput(0), ew96->getOutput(0), ew95->getOutput(0), ew94->getOutput(0)};
    IConcatenationLayer* concat100 = network->addConcatenation(input_tensor_100, 6);
    concat100->setAxis(0);
    IElementWiseLayer* ew101 = convBnSilu(network, weightMap, *concat100->getOutput(0), 512, 1, 1, 0, "model.101");

    IElementWiseLayer* ew102 = RepConv(network, weightMap, *ew75->getOutput(0), 256, 3, 1, "model.102");
    IElementWiseLayer* ew103 = RepConv(network, weightMap, *ew88->getOutput(0), 512, 3, 1, "model.103");
    IElementWiseLayer* ew104 = RepConv(network, weightMap, *ew101->getOutput(0), 1024, 3, 1, "model.104");

    // out
    IConvolutionLayer* cv105_0 = network->addConvolutionNd(*ew102->getOutput(0), 3*(NUM_CLASS+5), DimsHW{1, 1}, weightMap["model.105.m.0.weight"], weightMap["model.105.m.0.bias"]);
    assert(cv105_0);
    cv105_0->setName("cv105.0");
    IConvolutionLayer* cv105_1 = network->addConvolutionNd(*ew103->getOutput(0), 3*(NUM_CLASS+5), DimsHW{1, 1}, weightMap["model.105.m.1.weight"], weightMap["model.105.m.1.bias"]);
    assert(cv105_1);
    cv105_1->setName("cv105.1");
    IConvolutionLayer* cv105_2 = network->addConvolutionNd(*ew104->getOutput(0), 3*(NUM_CLASS+5), DimsHW{1, 1}, weightMap["model.105.m.2.weight"], weightMap["model.105.m.2.bias"]);
    assert(cv105_2);
    cv105_2->setName("cv105.2");

    IShuffleLayer* sf0 = network->addShuffle(*cv105_0->getOutput(0));
    sf0->setReshapeDimensions(Dims4{3, NUM_CLASS+5, INPUT_H/8, INPUT_W/8});
    sf0->setSecondTranspose(Permutation{0, 2, 3, 1});
    IActivationLayer* act0 = network->addActivation(*sf0->getOutput(0), ActivationType::kSIGMOID);
    IShuffleLayer* out0 = network->addShuffle(*act0->getOutput(0));
    out0->setReshapeDimensions(DimsHW{3*INPUT_H/8*INPUT_W/8, NUM_CLASS+5});

    IShuffleLayer* sf1 = network->addShuffle(*cv105_1->getOutput(0));
    sf1->setReshapeDimensions(Dims4{3, NUM_CLASS+5, INPUT_H/16, INPUT_W/16});
    sf1->setSecondTranspose(Permutation{0, 2, 3, 1});
    IActivationLayer* act1 = network->addActivation(*sf1->getOutput(0), ActivationType::kSIGMOID);
    IShuffleLayer* out1 = network->addShuffle(*act1->getOutput(0));
    out1->setReshapeDimensions(DimsHW{3*INPUT_H/16*INPUT_W/16, NUM_CLASS+5});

    IShuffleLayer* sf2 = network->addShuffle(*cv105_2->getOutput(0));
    sf2->setReshapeDimensions(Dims4{3, NUM_CLASS+5, INPUT_H/32, INPUT_W/32});
    sf2->setSecondTranspose(Permutation{0, 2, 3, 1});
    IActivationLayer* act2 = network->addActivation(*sf2->getOutput(0), ActivationType::kSIGMOID);
    IShuffleLayer* out2 = network->addShuffle(*act2->getOutput(0));
    out2->setReshapeDimensions(DimsHW{3*INPUT_H/32*INPUT_W/32, 85});

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
        if(row[5] > 0.35){
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
