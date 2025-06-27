#!/bin/sh

c++ -O3 ffmpeg-read.cpp -o ffmpeg-read -I/usr/include/ffmpeg -I/usr/include/opencv4 -fpermissive -lavcodec -lavutil -lavfilter -lavformat -lopencv_core -lopencv_highgui -lopencv_imgproc
