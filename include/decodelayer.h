#ifndef YOLOV7_DECODELAYER_H
#define YOLOV7_DECODELAYER_H

#include <string>
#include <vector>
#include <NvInfer.h>


namespace nvinfer1{

    class DecodeLayerPlugin: public IPluginV2DynamicExt{
    public:
        DecodeLayerPlugin(int classCount, int kptCount, int maxOutObject, std::vector<float> anchors);
        DecodeLayerPlugin(const void* data, size_t length);
        ~DecodeLayerPlugin() noexcept;

        int getNbOutputs() const override{ return 1;}

//        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
       virtual DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;

        int initialize() override;

        virtual void terminate()  override {};

//        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
        size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override { return 0; }

//        virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
        int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void* buffer) const override;

//        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const  override {
//                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
//        }
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const  override;

        const char* getPluginVersion() const override;

        void destroy() override;

//        IPluginV2IOExt* clone() const override;
        IPluginV2DynamicExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

//        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const  override;

//        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

//        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
        void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;

        void detachFromContext() override;

    private:
        void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize, int firstHeight, int firstWidth);
        int mThreadCount = 256;
        int mClassCount;
        int mKptCount;
        int mAnchorLen;
        int mMaxOutObject;
        void* mAnchor_h;
        void* mAnchor_d;
        const char* mPluginNamespace;
    };


    class DecodePluginCreator: public IPluginCreator{
    public:
        DecodePluginCreator();

        ~DecodePluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const PluginFieldCollection* getFieldNames() override;

//        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
        IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

//        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
        IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

        void setPluginNamespace(const char* libNamespace) override{ mNamespace = libNamespace;}

        const char* getPluginNamespace() const override{ return mNamespace.c_str();}

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };

    REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
}

#endif //YOLOV7_DECODELAYER_H
