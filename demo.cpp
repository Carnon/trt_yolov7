#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov7.h"

void drawResult(float* result, int num, cv::Mat &frame){
    for(int i=0; i<num; i++){
        float x = result[6*i + 0];
        float y = result[6*i + 1];
        float w = result[6*i + 2];
        float h = result[6*i + 3];
        std::string name = std::to_string(int(result[6*i+4]));
        float conf = result[6*i + 5];

        cv::putText(frame, name, cv::Point(int(std::max(0.0f, x-w/2)), int(std::max(y-h/2-5.0, 0.0))), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::Rect rst(int(x - w/2), int(y-h/2), int(w), int(h));
        cv::rectangle(frame, rst, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
    }
}


int main(int argc, char** argv){

    if(argc != 4){
        std::cout<<"params error !!  ./demo [-s] [wts_path] [engine_path]  or ./demo [-d] [engine_path] [image_path] "<<std::endl;
        return 0;
    }

    if(strcmp(argv[1], "-s") == 0){
        std::string wts_path = argv[2];
        std::string engine_path = argv[3];
        serialize(wts_path, engine_path);
    }else if(std::strcmp(argv[1], "-d") == 0){
        std::string engine_path = argv[2];
        std::string image_path = argv[3];

        loadEngine(engine_path.c_str());
        cv::Mat frame = cv::imread(image_path);
        uchar* data = frame.data;
        int w = frame.cols;
        int h = frame.rows;
        auto* result = new float[1000*6];
        auto start = std::chrono::high_resolution_clock::now();
        int num = inferImage((uint8_t *)data, w, h, result);
        auto end = std::chrono::high_resolution_clock::now();
        auto total = std::chrono::duration<float, std::milli>(end - start).count();
        std::cout<<"infer time: "<<total<<std::endl;

        drawResult(result, num, frame);
        cv::imwrite("result.jpg", frame);
        delete[] result;
        release();
    }else if(std::strcmp(argv[1], "-b") == 0){
        std::string engine_path = argv[2];
        std::string image_path = argv[3];
        loadEngine(engine_path.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        inferBatchImage(image_path.c_str());
        auto end = std::chrono::high_resolution_clock::now();
        auto total = std::chrono::duration<float, std::milli>(end - start).count();
        std::cout<<"infer time: "<<total<<std::endl;
        release();
    }else{
        std::cout<<"error params"<<std::endl;
    }

}

