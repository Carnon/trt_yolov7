#ifndef TRT_YOLOV7_H
#define TRT_YOLOV7_H

#include <iostream>
#include "common.hpp"
#include "logging.h"


void serialize(const std::string& wts_path, const std::string& engine_path);

#endif //TRT_YOLOV7_H
