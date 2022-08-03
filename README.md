# study yolov7

without onnx parser

ubuntu18.04 cuda-11.1 tensorrt-7.2.2.3 opencv

```
    git clone https://github.com/WongKinYiu/yolov7.git
    
    cp {trt_yolov7}/gen_wts.py {WongKinYiu}/yolov7
    
    cd {WongKinYiu}/yolov7
    
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
    
    python gen_wts.py
    
    cp {WongKinYiu}/yolov7-tiny.wts {trt_yolov7}/
    
    cd {trt_yolov7}
    
    vim cmakelists.txt   and  set tensort path; this depend opencv cuda tensorrt
    
    mkdir build
    
    cd build
    
    cmake ..
    
    make
    
    ./demo -s ../yolov7-tiny.wts ../yolov7-tiny.engine
    
    ./demo -d ../yolov7-tiny.engine ../image/test.jpeg
```