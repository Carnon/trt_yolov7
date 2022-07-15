//
// Created by carnon on 2022/6/20.
//

#include "preprocess.h"


__global__ void resize_img_kernel(uint8_t* src, int src_width, int src_height,
                                  float* dst, int dst_width, int dst_height,
                                  uint8_t const_value_st, float scale){

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if(position > dst_width* dst_height) return;

    int dx = position % dst_width;
    int dy = position / dst_width;

    float src_x = float(dx) / scale;
    float src_y = float(dy) / scale;


    float c0, c1, c2;
    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }else{
        int y_low = int(floorf(src_y));
        int x_low = int(floorf(src_x));
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[3] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - float(y_low);
        float lx = src_x - float(x_low);
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy*hx, w2 = hy*lx, w3 = ly*hx, w4 = ly*lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if(y_low >=0){
            if(x_low >=0){
                v1 = src + y_low * src_width * 3 + x_low * 3;
            }
            if(x_high < src_width)
                v2 = src + y_low * src_width * 3 + x_high * 3;
        }

        if(y_high< src_height){
            if(x_low >= 0){
                v3 = src + y_high * src_width * 3 + x_low * 3;
            }
            if(x_high < src_width){
                v4 = src + y_high * src_width * 3 + x_high * 3;
            }
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

//    float* pdst = dst+ dy*dst_width*3 + dx*3;
//    pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = float(c0) / 255.f;
    c1 = float(c1) / 255.f;
    c2 = float(c2) / 255.f;

    // channel last to channel first
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;

    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void resize_img(uint8_t* src, int src_width, int src_height,
                float* dst, int dst_width, int dst_height,
                float scale, cudaStream_t stream){

    int jobs = dst_height * dst_width;
    int thread = 256;

    int blocks = ceil(float(jobs) / float(thread));

    resize_img_kernel<<<blocks, thread, 0, stream>>>
    (src, src_width, src_height,
     dst, dst_width, dst_height,
     128, scale);
}

