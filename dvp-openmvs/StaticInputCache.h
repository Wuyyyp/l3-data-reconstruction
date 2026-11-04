#pragma once
#include <map>
#include <deque>
#include <functional>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
inline bool StaticCacheOn(){const char* v=std::getenv("DVP_STATIC_CACHE");return v && std::atoi(v);}
struct StaticCacheStore {
 std::map<std::string,cv::Mat> maps; std::deque<std::string> order;
 std::map<std::string,Camera> cameras;
 size_t bytes=0,hits=0,misses=0,verified=0,evictions=0;
};
inline StaticCacheStore& StaticStore(){static StaticCacheStore s;return s;}
inline bool ExactMat(const cv::Mat& a,const cv::Mat& b){
 if(a.size()!=b.size()||a.type()!=b.type())return false;
 for(int y=0;y<a.rows;++y)if(std::memcmp(a.ptr(y),b.ptr(y),a.cols*a.elemSize()))return false;
 return true;
}
inline cv::Mat StaticMat(const std::string& key,const std::function<cv::Mat()>& make){
 if(!StaticCacheOn())return make();auto& s=StaticStore();auto it=s.maps.find(key);
 if(it!=s.maps.end()){
  ++s.hits;
  if(std::getenv("DVP_CACHE_VERIFY")){cv::Mat original=make();if(!ExactMat(it->second,original))throw std::runtime_error("Cache mismatch: "+key);++s.verified;}
  return it->second.clone(); // preserve private writable ownership
 }
 ++s.misses;cv::Mat m=make();if(m.empty())throw std::runtime_error("Empty static cache input: "+key);
 const size_t n=m.total()*m.elemSize(),limit=8ULL*1024*1024*1024;
 while(s.bytes+n>limit && !s.order.empty()){
  auto victim=s.maps.find(s.order.front());s.bytes-=victim->second.total()*victim->second.elemSize();s.maps.erase(victim);s.order.pop_front();++s.evictions;
 }
 if(n<=limit){s.maps[key]=m.clone();s.order.push_back(key);s.bytes+=n;}
 return m;
}
inline cv::Mat StaticGrayFloat(const std::string& name){
 return StaticMat("gray:"+name,[&](){cv::Mat u=cv::imread(name,cv::IMREAD_GRAYSCALE),f;u.convertTo(f,CV_32FC1);return f;});
}
inline cv::Mat StaticSourceFloat(const std::string& name,int height,int width){
 return StaticMat("canvas:"+name+":"+std::to_string(height)+":"+std::to_string(width),[&](){
  cv::Mat image_float=StaticGrayFloat(name),result=cv::Mat::zeros(height,width,image_float.type());
  // Exactly the original padding/cropping loop on misses; no new interpolation.
  for(int i=0;i<height;++i)for(int j=0;j<width;++j)
   if(i<image_float.rows&&j<image_float.cols)result.at<float>(i,j)=image_float.at<float>(i,j);
  return result;
 });
}
inline bool StaticReadCamera(const path& file,Camera& cam){
 if(!StaticCacheOn())return ReadCamera(file,cam);auto& s=StaticStore();std::string k=file.string();auto it=s.cameras.find(k);
 if(it!=s.cameras.end()){
  ++s.hits;
  if(std::getenv("DVP_CACHE_VERIFY")){
   Camera fresh={};if(!ReadCamera(file,fresh))return false;
   const Camera& x=it->second;
   if(std::memcmp(x.K,fresh.K,sizeof(x.K))||std::memcmp(x.R,fresh.R,sizeof(x.R))||std::memcmp(x.t,fresh.t,sizeof(x.t))||std::memcmp(x.c,fresh.c,sizeof(x.c))||std::memcmp(&x.depth_min,&fresh.depth_min,2*sizeof(float)))throw std::runtime_error("Camera cache mismatch: "+k);
   ++s.verified;
  }
  cam=it->second;return true;
 }
 Camera fresh={};if(!ReadCamera(file,fresh))return false;++s.misses;s.cameras[k]=fresh;cam=fresh;return true;
}
inline cv::Mat StaticEdge(const path& file){return StaticMat("edge:"+file.string(),[&](){cv::Mat m;if(!ReadBinMat(file,m))throw std::runtime_error("Missing edge: "+file.string());return m;});}
inline cv::Mat StaticValidMask(const path& file,int width,int height){
 return StaticMat("mask:"+file.string()+":"+std::to_string(width)+":"+std::to_string(height),[&](){
  cv::Mat block=cv::imread(file.string(),cv::IMREAD_GRAYSCALE);if(block.empty())throw std::runtime_error("Cannot read mask: "+file.string());
  cv::Mat ignored,scaled,valid;cv::compare(block,128,ignored,cv::CMP_LT);ignored.convertTo(ignored,CV_32F,1.0/255.0);
  cv::resize(ignored,scaled,cv::Size(width,height),0,0,cv::INTER_AREA);cv::compare(scaled,0,valid,cv::CMP_EQ);return valid;
 });
}
inline void PrintStaticCacheStats(){auto& s=StaticStore();std::cout<<"STATIC_CACHE_STATS hits="<<s.hits<<" misses="<<s.misses<<" verified="<<s.verified<<" bytes="<<s.bytes<<" entries="<<s.maps.size()<<" camera_entries="<<s.cameras.size()<<" evictions="<<s.evictions<<std::endl;}
