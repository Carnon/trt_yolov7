#include "opencv2/opencv.hpp"
#include "yolov7.h"

int main(int argc, char** argv){

    if(argc != 4){
        std::cout<<"params error !!  ./demo [-s] [../yolov7-w6-pose.wts] [../yolov7-w6-pose.engine]  "<<std::endl;
        return 0;
    }

    if(strcmp(argv[1], "-s") == 0){
        std::string wts_path = argv[2];
        std::string engine_path = argv[3];
        serialize(wts_path, engine_path);
    }else{
        std::cout<<"error params"<<std::endl;
    }
}

