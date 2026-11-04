#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
inline void SaveSnapMat(const std::string& name,const cv::Mat& m){
 std::ofstream f(name,std::ios::binary);int h[3]={m.rows,m.cols,m.type()};f.write((char*)h,sizeof(h));
 for(int y=0;y<m.rows;++y)f.write((char*)m.ptr(y),m.cols*m.elemSize());
 if(!f)throw std::runtime_error("snapshot write failed: "+name);
}
inline cv::Mat LoadSnapMat(const std::string& name){
 std::ifstream f(name,std::ios::binary);int h[3]={};f.read((char*)h,sizeof(h));
 if(!f||h[0]<1||h[1]<1)throw std::runtime_error("snapshot header failed: "+name);
 cv::Mat m(h[0],h[1],h[2]);for(int y=0;y<m.rows;++y)f.read((char*)m.ptr(y),m.cols*m.elemSize());
 if(!f)throw std::runtime_error("snapshot read failed: "+name);return m;
}
inline bool SameSnapMat(const cv::Mat& a,const cv::Mat& b){
 if(a.size()!=b.size()||a.type()!=b.type())return false;
 for(int y=0;y<a.rows;++y)if(std::memcmp(a.ptr(y),b.ptr(y),a.cols*a.elemSize()))return false;return true;
}
