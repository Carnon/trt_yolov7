#include "assert.h"
#include <vector>
#include <iostream>
#include "decodelayer.h"
#include "cuda_utils.h"

namespace Tn
{
    template<typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

namespace nvinfer1{

    DecodeLayerPlugin::DecodeLayerPlugin(int classCount, int netWidth, int netHeight, int maxOutObject, std::vector<float> anchors) {
        mClassCount = classCount;
        mYoloV7NetHeight = netHeight;
        mYoloV7NetWidth = netWidth;
        mMaxOutObject = maxOutObject;
        mAnchorLen = (int)anchors.size();
        CUDA_CHECK(cudaMallocHost(&mAnchor_h, mAnchorLen*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(mAnchor_h, &anchors[0], mAnchorLen*sizeof(float), cudaMemcpyHostToHost));

        CUDA_CHECK(cudaMalloc(&mAnchor_d, mAnchorLen*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(mAnchor_d, mAnchor_h, mAnchorLen*sizeof(float ), cudaMemcpyHostToDevice));
    }

    DecodeLayerPlugin::~DecodeLayerPlugin() noexcept {
        CUDA_CHECK(cudaFreeHost(mAnchor_h));
        CUDA_CHECK(cudaFree(mAnchor_d));
    }

    DecodeLayerPlugin::DecodeLayerPlugin(const void *data, size_t length) {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mYoloV7NetWidth);
        read(d, mYoloV7NetHeight);
        read(d, mMaxOutObject);
        read(d, mAnchorLen);
        CUDA_CHECK(cudaMallocHost(&mAnchor_h, mAnchorLen*sizeof(float)));
        memcpy(mAnchor_h, d, mAnchorLen*sizeof(float));
        d += mAnchorLen*sizeof(float);

        CUDA_CHECK(cudaMalloc(&mAnchor_d, mAnchorLen*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(mAnchor_d, mAnchor_h, mAnchorLen*sizeof(float), cudaMemcpyHostToDevice));

        assert(d == a+length);
    }

    void DecodeLayerPlugin::serialize(void* buffer) const{
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mYoloV7NetWidth);
        write(d, mYoloV7NetHeight);
        write(d, mMaxOutObject);
        write(d, mAnchorLen);

        memcpy(d, mAnchor_h, mAnchorLen*sizeof(float));
        d += mAnchorLen*sizeof(float);

        assert(d == a + getSerializationSize());
    }

    size_t DecodeLayerPlugin::getSerializationSize() const {
        return sizeof(mClassCount)+sizeof(mThreadCount)+sizeof(mYoloV7NetWidth)+sizeof(mYoloV7NetHeight)+sizeof(mMaxOutObject)+sizeof(mAnchorLen)+mAnchorLen*sizeof(float);
    }

    int DecodeLayerPlugin::initialize() { return 0;}

    Dims DecodeLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims){
        int totalSize = mMaxOutObject * 6;
        return Dims3{totalSize+1, 1, 1};
    }

    void DecodeLayerPlugin::setPluginNamespace(const char* pluginNamespace){mPluginNamespace = pluginNamespace;}

    const char* DecodeLayerPlugin::getPluginNamespace() const {return mPluginNamespace;}

    // Return the DataType of the plugin output at the requested index
    DataType DecodeLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {return DataType::kFLOAT;}

    // Return true if output tensor is broadcast across a batch.
    bool DecodeLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {return false;}

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool DecodeLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const {return false;}

    void DecodeLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) {}

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void DecodeLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) {}

    // Detach the plugin object from its execution context.
    void DecodeLayerPlugin::detachFromContext() {}

    const char* DecodeLayerPlugin::getPluginType() const {return "DecodeLayer_TRT";}

    const char* DecodeLayerPlugin::getPluginVersion() const {return "1";}

    void DecodeLayerPlugin::destroy() { delete this;}

    // Clone the plugin
    IPluginV2IOExt* DecodeLayerPlugin::clone() const {
        std::vector<float> anchors((float*)mAnchor_h, (float*)mAnchor_h+mAnchorLen);
        auto* p = new DecodeLayerPlugin(mClassCount, mYoloV7NetWidth, mYoloV7NetHeight, mMaxOutObject,  anchors);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }


    __global__ void calDetection(const float *input, float* output, int noElements, const int input_w, const int input_h,
                                 int maxOutObject, const int yoloWidth, const int yoloHeight, const float* yoloAnchor,
                                 int classes, int outputElem){

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx > noElements) return;

        int totalGrid = yoloWidth * yoloHeight;
        int bnIdx = idx / totalGrid;
        idx = idx - totalGrid*bnIdx;

        int info_len_i = 5+classes;

        const float* curInput = input + bnIdx * (info_len_i * totalGrid * 3);

        for(int k=0; k<3; k++){
            float box_prob = curInput[idx + k*info_len_i*totalGrid + 4*totalGrid];

            if(box_prob < 0.1) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;

            for(int i=5; i<info_len_i; i++){
                float p = curInput[idx + k * info_len_i * totalGrid + i * totalGrid];

                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }

            float *res_count = output + bnIdx*outputElem;
            int count = int(atomicAdd(res_count, 1));

            if(count> maxOutObject) return;
            float *data = res_count + 1 + count*6;

            int y = idx / yoloWidth;
            int x = idx % yoloWidth;

            data[0] = (float(x) - 0.5f + 2* curInput[idx+k*info_len_i*totalGrid + 0 * totalGrid]) * float(input_w) / float(yoloWidth);
            data[1] = (float(y) - 0.5f + 2* curInput[idx+k*info_len_i*totalGrid + 1 * totalGrid]) * float(input_h) / float(yoloHeight);
            data[2] = pow(2.0f * curInput[idx + k*info_len_i * totalGrid + 2 * totalGrid], 2) * float(yoloAnchor[2*k]);
            data[3] = pow(2.0f * curInput[idx + k*info_len_i * totalGrid + 3 * totalGrid], 2) * float(yoloAnchor[2*k+1]);
            data[4] = float(class_id);
            data[5] = box_prob * max_cls_prob;
        }
    }

    void DecodeLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize){
        int outputElem = 1 + mMaxOutObject * 6;
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
        }

        for(int i=0; i< 3; i++){
            int scale = 8 * int(pow(2, i));
            int yoloWidth = mYoloV7NetWidth / scale;
            int yoloHeight = mYoloV7NetHeight / scale;
            int numElem = yoloHeight * yoloWidth * batchSize;
            const float* anchor_d = (float*)mAnchor_d + 6*i;
            calDetection<<<(numElem+mThreadCount-1)/mThreadCount, mThreadCount, 0, stream>>>
            (inputs[i], output, numElem, mYoloV7NetWidth, mYoloV7NetHeight, mMaxOutObject, yoloWidth, yoloHeight, anchor_d, mClassCount, outputElem);
        }
    }


    int DecodeLayerPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream){
        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection DecodePluginCreator::mFC{};
    std::vector<PluginField> DecodePluginCreator::mPluginAttributes;

    // creator
    DecodePluginCreator::DecodePluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* DecodePluginCreator::getPluginName() const{return "DecodeLayer_TRT";}

    const char* DecodePluginCreator::getPluginVersion() const {return "1"; }

    const PluginFieldCollection* DecodePluginCreator::getFieldNames() {return &mFC;}

    IPluginV2IOExt* DecodePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc){
        assert(fc->nbFields == 2);
        assert(strcmp(fc->fields[0].name, "netInfo") == 0);
        assert(strcmp(fc->fields[1].name, "anchor") == 0);
        int *p_netInfo = (int*)(fc->fields[0].data);
        int class_count = p_netInfo[0];
        int input_w = p_netInfo[1];
        int input_h = p_netInfo[2];
        int max_output_object_count = p_netInfo[3];

        std::vector<float> anchor((float*)fc->fields[1].data, (float*)fc->fields[1].data + fc->fields[1].length);
        auto* obj = new DecodeLayerPlugin(class_count, input_w, input_h, max_output_object_count, anchor);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* DecodePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength){
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        DecodeLayerPlugin* obj = new DecodeLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}

