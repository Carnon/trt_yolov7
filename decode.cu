#include "decode.h"

__global__ void decode_output_kernel(float* src, float* dst, int input_width, int input_height, int NUM_CLASS, float scale, int jobs){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if(position >= jobs) return;

//    int anchors[3][3][2] = {{{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
    int anchors[3][3][2] = {{{12, 16}, {19, 36}, {40, 28}}, {{36, 75}, {76, 55}, {72, 146}}, {{142, 110}, {192, 243}, {459, 401}}};
    int grids[3][3] = {{3, 80, 80}, {3, 40, 40}, {3, 20, 20}};

    int a, c, x, y;

    int i=position;

    if(i < 80*80){
        a = 0;
        c = 0;
        x = i % 80;
        y = i / 80;
    }else if(i < 80*80*2){
        a = 0;
        c = 1;
        x = (i - 80*80) % 80;
        y = (i - 80*80) / 80;
    }else if(i < 80*80*3){
        a = 0;
        c = 2;
        x = (i - 80*80*2) % 80;
        y = (i - 80*80*2) / 80;

    }else if(i < 80*80*3+40*40){
        a = 1;
        c = 0;
        x = (i - 80*80*3) % 40;
        y = (i - 80*80*3) / 40;
    }else if(i < 80*80*3+40*40*2){
        a = 1;
        c = 1;
        x = (i - 80*80*3 - 40*40) % 40;
        y = (i - 80*80*3 - 40*40) / 40;
    }else if(i < 80*80*3 + 40*40*3){
        a = 1;
        c = 2;
        x = (i - 80*80*3 - 40*40*2) % 40;
        y = (i - 80*80*3 - 40*40*2) / 40;
    }else if(i < 80*80*3 + 40*40*3 + 20*20){
        a = 2;
        c = 0;
        x = (i - 80*80*3 - 40*40*3) % 20;
        y = (i - 80*80*3 - 40*40*3) / 20;
    }else if(i < 80*80*3 + 40*40*3 + 20*20*2){
        a = 2;
        c = 1;
        x = (i - 80*80*3 - 40*40*3 - 20*20) % 20;
        y = (i - 80*80*3 - 40*40*3 - 20*20) / 20;
    }else{
        a = 2;
        c = 2;
        x = (i - 80*80*3 - 40*40*3 - 20*20*2) % 20;
        y = (i - 80*80*3 - 40*40*3 - 20*20*2) / 20;
    }

    float *row = src + (NUM_CLASS+5)*i;
    int class_id = 0;
    float max_cls_prob = 0.0;

    for(int j=5; j<NUM_CLASS+5; j++){
        if(row[j] > max_cls_prob){
            max_cls_prob = row[j];
            class_id = j - 5;
        }
    }

    float conf = row[4] * max_cls_prob;

    float box_x = (row[0]*2 - 0.5f + float(x)) / float(grids[a][1]) * float(input_width) * scale;
    float box_y = (row[1]*2 - 0.5f + float(y)) / float(grids[a][2]) * float(input_height) * scale;
    float box_w = pow(row[2]*2, 2) * float(anchors[a][c][0]) * scale;
    float box_h = pow(row[3]*2, 2) * float(anchors[a][c][1]) * scale;

    dst[i*6+0] = box_x;
    dst[i*6+1] = box_y;
    dst[i*6+2] = box_w;
    dst[i*6+3] = box_h;
    dst[i*6+4] = float(class_id);
    dst[i*6+5] = conf;
}


void decode_output(float* src, float* dst, int input_width, int input_height, int num_class, float scale, cudaStream_t stream){
    int jobs = 80*80*3 + 40*40*3 + 20*20*3;
    int thread = 256;

    int blocks = (jobs+256-1) / thread;
    decode_output_kernel<<<blocks, thread, 0, stream>>>
    (src, dst, input_width, input_height, num_class, scale, jobs);
}
