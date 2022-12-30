#include <algorithm>
#include <opencv/opencv.hpp>
#include "cuda_utils.h"
#include "yolov7.h"

#define DEVICE 0

static Logger gLogger;

static const int INPUT_W = -1;
static const int INPUT_H = -1;
static const int NUM_CLASS = 1;
static const int NUM_KPT = 17;
static const int BATCH_SIZE = 1;
static const int MAX_INPUT_SIZE = 1024;
static const int MIN_INPUT_SIZE = 256;
static const int OPT_INPUT_SIZE = 960;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const int MAX_OBJECT_SIZE = 1000;
const int MAX_IMAGE_INPUT_SIZE = 3000*3000;


ICudaEngine* build_pose_engine(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path){
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(1U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);

    // backbone
    IConcatenationLayer* conv0 = reOrg(network, weightMap, *data);

    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 64, 3, 1, 1, "model.1");
    IElementWiseLayer* conv2 = convBnSilu(network, weightMap, *conv1->getOutput(0), 128, 3, 2, 1, "model.2");
    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.3");
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.4");
    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv4->getOutput(0), 64, 3, 1, 1, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");
    ITensor* input_tensor_9[] = {conv8->getOutput(0), conv6->getOutput(0), conv4->getOutput(0), conv3->getOutput(0)};
    IConcatenationLayer* cat9 = network->addConcatenation(input_tensor_9, 4);

    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *cat9->getOutput(0), 128, 1, 1, 0, "model.10");
    IElementWiseLayer* conv11 = convBnSilu(network, weightMap, *conv10->getOutput(0), 256, 3, 2, 1, "model.11");
    IElementWiseLayer* conv12 = convBnSilu(network, weightMap, *conv11->getOutput(0), 128, 1, 1, 0, "model.12");
    IElementWiseLayer* conv13 = convBnSilu(network, weightMap, *conv11->getOutput(0), 128, 1, 1, 0, "model.13");
    IElementWiseLayer* conv14 = convBnSilu(network, weightMap, *conv13->getOutput(0), 128, 3, 1, 1, "model.14");
    IElementWiseLayer* conv15 = convBnSilu(network, weightMap, *conv14->getOutput(0), 128, 3, 1, 1, "model.15");
    IElementWiseLayer* conv16 = convBnSilu(network, weightMap, *conv15->getOutput(0), 128, 3, 1, 1, "model.16");
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *conv16->getOutput(0), 128, 3, 1, 1, "model.17");
    ITensor* input_tensor_18[] = {conv17->getOutput(0), conv15->getOutput(0), conv13->getOutput(0), conv12->getOutput(0)};
    IConcatenationLayer* cat18 = network->addConcatenation(input_tensor_18, 4);

    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *cat18->getOutput(0), 256, 1, 1, 0, "model.19");
    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *conv19->getOutput(0), 512, 3, 2, 1, "model.20");
    IElementWiseLayer* conv21 = convBnSilu(network, weightMap, *conv20->getOutput(0), 256, 1, 1, 0, "model.21");
    IElementWiseLayer* conv22 = convBnSilu(network, weightMap, *conv20->getOutput(0), 256, 1, 1, 0, "model.22");
    IElementWiseLayer* conv23 = convBnSilu(network, weightMap, *conv22->getOutput(0), 256, 3, 1, 1, "model.23");
    IElementWiseLayer* conv24 = convBnSilu(network, weightMap, *conv23->getOutput(0), 256, 3, 1, 1, "model.24");
    IElementWiseLayer* conv25 = convBnSilu(network, weightMap, *conv24->getOutput(0), 256, 3, 1, 1, "model.25");
    IElementWiseLayer* conv26 = convBnSilu(network, weightMap, *conv25->getOutput(0), 256, 3, 1, 1, "model.26");
    ITensor* input_tensor_27[] = {conv26->getOutput(0), conv24->getOutput(0), conv22->getOutput(0), conv21->getOutput(0)};
    IConcatenationLayer* cat27 = network->addConcatenation(input_tensor_27, 4);

    IElementWiseLayer* conv28 = convBnSilu(network, weightMap, *cat27->getOutput(0), 512, 1, 1, 0, "model.28");
    IElementWiseLayer* conv29 = convBnSilu(network, weightMap, *conv28->getOutput(0), 768, 3, 2, 1, "model.29");
    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *conv29->getOutput(0), 384, 1, 1, 0, "model.30");
    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *conv29->getOutput(0), 384, 1, 1, 0, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 384, 3, 1, 1, "model.32");
    IElementWiseLayer* conv33 = convBnSilu(network, weightMap, *conv32->getOutput(0), 384, 3, 1, 1, "model.33");
    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *conv33->getOutput(0), 384, 3, 1, 1, "model.34");
    IElementWiseLayer* conv35 = convBnSilu(network, weightMap, *conv34->getOutput(0), 384, 3, 1, 1, "model.35");
    ITensor* input_tensor_36[] = {conv35->getOutput(0), conv33->getOutput(0), conv31->getOutput(0), conv30->getOutput(0)};
    IConcatenationLayer* cat36 = network->addConcatenation(input_tensor_36, 4);

    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *cat36->getOutput(0), 768, 1, 1, 0, "model.37");
    IElementWiseLayer* conv38 = convBnSilu(network, weightMap, *conv37->getOutput(0), 1024, 3, 2, 1, "model.38");
    IElementWiseLayer* conv39 = convBnSilu(network, weightMap, *conv38->getOutput(0), 512, 1, 1, 0, "model.39");
    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *conv38->getOutput(0), 512, 1, 1, 0, "model.40");
    IElementWiseLayer* conv41 = convBnSilu(network, weightMap, *conv40->getOutput(0), 512, 3, 1, 1, "model.41");
    IElementWiseLayer* conv42 = convBnSilu(network, weightMap, *conv41->getOutput(0), 512, 3, 1, 1, "model.42");
    IElementWiseLayer* conv43 = convBnSilu(network, weightMap, *conv42->getOutput(0), 512, 3, 1, 1, "model.43");
    IElementWiseLayer* conv44 = convBnSilu(network, weightMap, *conv43->getOutput(0), 512, 3, 1, 1, "model.44");
    ITensor* input_tensor_45[] = {conv44->getOutput(0), conv42->getOutput(0), conv40->getOutput(0), conv39->getOutput(0)};
    IConcatenationLayer* cat45 = network->addConcatenation(input_tensor_45, 4);

    IElementWiseLayer* conv46 = convBnSilu(network, weightMap, *cat45->getOutput(0), 1024, 1, 1, 0, "model.46");

    // head
    IElementWiseLayer* conv47 = SPPCSPC(network, weightMap, *conv46->getOutput(0), 512, "model.47");
    IElementWiseLayer* conv48 = convBnSilu(network, weightMap, *conv47->getOutput(0), 384, 1, 1, 0, "model.48");
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    IResizeLayer* re49 = network->addResize(*conv48->getOutput(0));
    re49->setResizeMode(ResizeMode::kNEAREST);
    re49->setScales(scale, 4);
    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *conv37->getOutput(0), 384, 1, 1, 0, "model.50");
    ITensor* input_tensor_51[] = {conv50->getOutput(0), re49->getOutput(0)};
    IConcatenationLayer* cat51 = network->addConcatenation(input_tensor_51, 2);
    IElementWiseLayer* conv52 = convBnSilu(network, weightMap, *cat51->getOutput(0), 384, 1, 1, 0, "model.52");
    IElementWiseLayer* conv53 = convBnSilu(network, weightMap, *cat51->getOutput(0), 384, 1, 1, 0, "model.53");
    IElementWiseLayer* conv54 = convBnSilu(network, weightMap, *conv53->getOutput(0), 192, 3, 1, 1, "model.54");
    IElementWiseLayer* conv55 = convBnSilu(network, weightMap, *conv54->getOutput(0), 192, 3, 1, 1, "model.55");
    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *conv55->getOutput(0), 192, 3, 1, 1, "model.56");
    IElementWiseLayer* conv57 = convBnSilu(network, weightMap, *conv56->getOutput(0), 192, 3, 1, 1, "model.57");
    ITensor* input_tensor_58[] = {conv57->getOutput(0), conv56->getOutput(0), conv55->getOutput(0), conv54->getOutput(0), conv53->getOutput(0), conv52->getOutput(0)};
    IConcatenationLayer* cat58 = network->addConcatenation(input_tensor_58, 6);

    IElementWiseLayer* conv59 = convBnSilu(network, weightMap, *cat58->getOutput(0), 384, 1, 1, 0, "model.59");
    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv59->getOutput(0), 256, 1, 1, 0, "model.60");
    IResizeLayer* re61 = network->addResize(*conv60->getOutput(0));
    re61->setResizeMode(ResizeMode::kNEAREST);
    re61->setScales(scale, 4);
    IElementWiseLayer* conv62 = convBnSilu(network, weightMap, *conv28->getOutput(0), 256, 1, 1, 0, "model.62");
    ITensor* input_tensor_63[] = {conv62->getOutput(0), re61->getOutput(0)};
    IConcatenationLayer* cat63 = network->addConcatenation(input_tensor_63, 2);
    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *cat63->getOutput(0), 256, 1, 1, 0, "model.64");
    IElementWiseLayer* conv65 = convBnSilu(network, weightMap, *cat63->getOutput(0), 256, 1, 1, 0, "model.65");
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *conv65->getOutput(0), 128, 3, 1, 1, "model.66");
    IElementWiseLayer* conv67 = convBnSilu(network, weightMap, *conv66->getOutput(0), 128, 3, 1, 1, "model.67");
    IElementWiseLayer* conv68 = convBnSilu(network, weightMap, *conv67->getOutput(0), 128, 3, 1, 1, "model.68");
    IElementWiseLayer* conv69 = convBnSilu(network, weightMap, *conv68->getOutput(0), 128, 3, 1, 1, "model.69");
    ITensor* input_tensor_70[] = {conv69->getOutput(0), conv68->getOutput(0), conv67->getOutput(0), conv66->getOutput(0), conv65->getOutput(0), conv64->getOutput(0)};
    IConcatenationLayer* cat70 = network->addConcatenation(input_tensor_70, 6);

    IElementWiseLayer* conv71 = convBnSilu(network, weightMap, *cat70->getOutput(0), 256, 1, 1, 0, "model.71");
    IElementWiseLayer* conv72 = convBnSilu(network, weightMap, *conv71->getOutput(0), 128, 1, 1, 0, "model.72");
    IResizeLayer* re73 = network->addResize(*conv72->getOutput(0));
    re73->setScales(scale, 4);
    re73->setResizeMode(ResizeMode::kNEAREST);
    IElementWiseLayer* conv74 = convBnSilu(network, weightMap, *conv19->getOutput(0), 128, 1, 1, 0, "model.74");
    ITensor* input_tensor_75[] = {conv74->getOutput(0), re73->getOutput(0)};
    IConcatenationLayer* cat75 = network->addConcatenation(input_tensor_75, 2);
    IElementWiseLayer* conv76 = convBnSilu(network, weightMap, *cat75->getOutput(0), 128, 1, 1, 0, "model.76");
    IElementWiseLayer* conv77 = convBnSilu(network, weightMap, *cat75->getOutput(0), 128, 1, 1, 0, "model.77");
    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *conv77->getOutput(0), 64, 3, 1, 1, "model.78");
    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *conv78->getOutput(0), 64, 3, 1, 1, "model.79");
    IElementWiseLayer* conv80 = convBnSilu(network, weightMap, *conv79->getOutput(0), 64, 3, 1, 1, "model.80");
    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *conv80->getOutput(0), 64, 3, 1, 1, "model.81");
    ITensor* input_tensor_82[] = {conv81->getOutput(0), conv80->getOutput(0), conv79->getOutput(0), conv78->getOutput(0), conv77->getOutput(0), conv76->getOutput(0)};
    IConcatenationLayer* cat82 = network->addConcatenation(input_tensor_82, 6);

    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *cat82->getOutput(0), 128, 1, 1, 0, "model.83");
    IElementWiseLayer* conv84 = convBnSilu(network, weightMap, *conv83->getOutput(0), 256, 3, 2, 1, "model.84");
    ITensor* input_tensor_85[] = {conv84->getOutput(0), conv71->getOutput(0)};
    IConcatenationLayer* cat85 = network->addConcatenation(input_tensor_85, 2);
    IElementWiseLayer* conv86 = convBnSilu(network, weightMap, *cat85->getOutput(0), 256, 1, 1, 0, "model.86");
    IElementWiseLayer* conv87 = convBnSilu(network, weightMap, *cat85->getOutput(0), 256, 1, 1, 0, "model.87");
    IElementWiseLayer* conv88 = convBnSilu(network, weightMap, *conv87->getOutput(0), 128, 3, 1, 1, "model.88");
    IElementWiseLayer* conv89 = convBnSilu(network, weightMap, *conv88->getOutput(0), 128, 3, 1, 1, "model.89");
    IElementWiseLayer* conv90 = convBnSilu(network, weightMap, *conv89->getOutput(0), 128, 3, 1, 1, "model.90");
    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *conv90->getOutput(0), 128, 3, 1, 1, "model.91");
    ITensor* input_tensor_92[] = {conv91->getOutput(0), conv90->getOutput(0), conv89->getOutput(0), conv88->getOutput(0), conv87->getOutput(0), conv86->getOutput(0)};
    IConcatenationLayer* cat92 = network->addConcatenation(input_tensor_92, 6);

    IElementWiseLayer* conv93 = convBnSilu(network, weightMap, *cat92->getOutput(0), 256, 1, 1, 0, "model.93");
    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *conv93->getOutput(0), 384, 3, 2, 1, "model.94");
    ITensor* input_tensor_95[] = {conv94->getOutput(0), conv59->getOutput(0)};
    IConcatenationLayer* cat95 = network->addConcatenation(input_tensor_95, 2);
    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *cat95->getOutput(0), 384, 1, 1, 0, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *cat95->getOutput(0), 384, 1, 1, 0, "model.97");
    IElementWiseLayer* conv98 = convBnSilu(network, weightMap, *conv97->getOutput(0), 192, 3, 1, 1, "model.98");
    IElementWiseLayer* conv99 = convBnSilu(network, weightMap, *conv98->getOutput(0), 192, 3, 1, 1, "model.99");
    IElementWiseLayer* conv100 = convBnSilu(network, weightMap, *conv99->getOutput(0), 192, 3, 1, 1, "model.100");
    IElementWiseLayer* conv101 = convBnSilu(network, weightMap, *conv100->getOutput(0), 192, 3, 1, 1, "model.101");
    ITensor* input_tensor_102[] = {conv101->getOutput(0), conv100->getOutput(0), conv99->getOutput(0), conv98->getOutput(0), conv97->getOutput(0), conv96->getOutput(0)};
    IConcatenationLayer* cat102 = network->addConcatenation(input_tensor_102, 6);

    IElementWiseLayer* conv103 = convBnSilu(network, weightMap, *cat102->getOutput(0), 384, 1, 1, 0, "model.103");
    IElementWiseLayer* conv104 = convBnSilu(network, weightMap, *conv103->getOutput(0), 512, 3, 2, 1, "model.104");
    ITensor* input_tensor_105[] = {conv104->getOutput(0), conv47->getOutput(0)};
    IConcatenationLayer* cat105 = network->addConcatenation(input_tensor_105, 2);
    IElementWiseLayer* conv106 = convBnSilu(network, weightMap, *cat105->getOutput(0), 512, 1, 1, 0, "model.106");
    IElementWiseLayer* conv107 = convBnSilu(network, weightMap, *cat105->getOutput(0), 512, 1, 1, 0, "model.107");
    IElementWiseLayer* conv108 = convBnSilu(network, weightMap, *conv107->getOutput(0), 256, 3, 1, 1, "model.108");
    IElementWiseLayer* conv109 = convBnSilu(network, weightMap, *conv108->getOutput(0), 256, 3, 1, 1, "model.109");
    IElementWiseLayer* conv110 = convBnSilu(network, weightMap, *conv109->getOutput(0), 256, 3, 1, 1, "model.110");
    IElementWiseLayer* conv111 = convBnSilu(network, weightMap, *conv110->getOutput(0), 256, 3, 1, 1, "model.111");
    ITensor* input_tensor_112[] = {conv111->getOutput(0), conv110->getOutput(0), conv109->getOutput(0), conv108->getOutput(0), conv107->getOutput(0), conv106->getOutput(0)};
    IConcatenationLayer* cat112 = network->addConcatenation(input_tensor_112, 6);

    IElementWiseLayer* conv113 = convBnSilu(network, weightMap, *cat112->getOutput(0), 512, 1, 1, 0, "model.113");

    IElementWiseLayer* conv114 = convBnSilu(network, weightMap, *conv83->getOutput(0), 256, 3, 1, 1, "model.114");
    IElementWiseLayer* conv115 = convBnSilu(network, weightMap, *conv93->getOutput(0), 512, 3, 1, 1, "model.115");
    IElementWiseLayer* conv116 = convBnSilu(network, weightMap, *conv103->getOutput(0), 768, 3, 1, 1, "model.116");
    IElementWiseLayer* conv117 = convBnSilu(network, weightMap, *conv113->getOutput(0), 1024, 3, 1, 1, "model.117");

    // output
    IElementWiseLayer* det0 = det(network, weightMap, *conv114->getOutput(0), 256, NUM_CLASS, "0");
    IConvolutionLayer* kpt0 = kpt(network, weightMap, *conv114->getOutput(0), 256, NUM_KPT, "model.118.m_kpt.0");
    ITensor* input_tensor_0[] = {det0->getOutput(0), kpt0->getOutput(0)};
    IConcatenationLayer* out0 = network->addConcatenation(input_tensor_0, 2);

    IElementWiseLayer* det1 = det(network, weightMap, *conv115->getOutput(0), 512, NUM_CLASS, "1");
    IConvolutionLayer* kpt1 = kpt(network, weightMap, *conv115->getOutput(0), 512, NUM_KPT, "model.118.m_kpt.1");
    ITensor* input_tensor_1[] = {det1->getOutput(0), kpt1->getOutput(0)};
    IConcatenationLayer* out1 = network->addConcatenation(input_tensor_1, 2);

    IElementWiseLayer* det2 = det(network, weightMap, *conv116->getOutput(0), 768, NUM_CLASS, "2");
    IConvolutionLayer* kpt2 = kpt(network, weightMap, *conv116->getOutput(0), 768, NUM_KPT, "model.118.m_kpt.2");
    ITensor* input_tensor_2[] = {det2->getOutput(0), kpt2->getOutput(0)};
    IConcatenationLayer* out2 = network->addConcatenation(input_tensor_2, 2);

    IElementWiseLayer* det3 = det(network, weightMap, *conv117->getOutput(0), 1024, NUM_CLASS, "3");
    IConvolutionLayer* kpt3 = kpt(network, weightMap, *conv117->getOutput(0), 1024, NUM_KPT, "model.118.m_kpt.3");
    ITensor* input_tensor_3[] = {det3->getOutput(0), kpt3->getOutput(0)};
    IConcatenationLayer* out3 = network->addConcatenation(input_tensor_3, 2);

    IPluginV2Layer* output = addDecodeLayer(network, weightMap, std::vector<IConcatenationLayer*>{out0, out1, out2, out3}, 4, NUM_CLASS, NUM_KPT, MAX_OBJECT_SIZE, "model.118");
    output->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*output->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, OPT_INPUT_SIZE, OPT_INPUT_SIZE));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
    config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize((1<<30));
    config->setFlag(BuilderFlag::kFP16);

    std::cout << "Building engine, please wait for a while... about ten years " << std::endl;
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
    ICudaEngine* engine = build_pose_engine(builder, config, DataType::kFLOAT, wts_path);
    IHostMemory* modelStream = engine->serialize();
    assert(modelStream != nullptr);
    std::ofstream p(engine_path, std::ios::binary);
    p.write(static_cast<const char *>(modelStream->data()), modelStream->size());

    modelStream->destroy();
    engine->destroy();
    builder->destroy();
    std::cout<<"convert ok!"<<std::endl;
}
