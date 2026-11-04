#pragma once
#include <functional>
#include <chrono>
#include <map>
struct P3Output {cv::Mat depth,normal,weak,views,radius;};
struct P3Packed {P3Output out;std::vector<cv::Mat> masks;};
template<class Engine> P3Packed PackP3(Engine& engine,const Problem& problem,const cv::Mat& valid_mask,const std::function<int()>& draw){
 int width=engine.GetWidth(),height=engine.GetHeight();
 P3Output out;out.depth=cv::Mat(height,width,CV_32FC1);out.normal=cv::Mat(height,width,CV_32FC3);out.weak=engine.GetPixelStates();out.views=engine.GetSelectedViews();out.radius=engine.GetRadiusMap();
 std::vector<cv::Mat> masks;
 for(int r=0;r<height;++r)for(int c=0;c<width;++c){
  auto plane=engine.GetPlaneHypothesis(r,c);bool keep=valid_mask.at<uchar>(r,c)>=128;
  out.depth.at<float>(r,c)=keep?plane.w:0.0f;
  if(!keep||out.depth.at<float>(r,c)<engine.GetDepthMin()||out.depth.at<float>(r,c)>engine.GetDepthMax()){out.depth.at<float>(r,c)=0;out.weak.at<uchar>(r,c)=UNKNOWN;}
  out.normal.at<cv::Vec3f>(r,c)=keep?cv::Vec3f(plane.x,plane.y,plane.z):cv::Vec3f(0,0,0);

 }
 return {out,masks};
}
template<class Engine> P3Output FinishP3(Engine& engine,const Problem& problem,const cv::Mat& valid_mask,P3Packed packed,const std::function<int()>& draw,const std::function<void(const char*,double)>& report={}){
 std::map<std::string,double> times;auto clock=std::chrono::steady_clock::now();auto fine=[&](const char* key){auto now=std::chrono::steady_clock::now();times[key]+=std::chrono::duration<double>(now-clock).count();clock=now;};
 int width=engine.GetWidth(),height=engine.GetHeight();auto out=packed.out;auto masks=std::move(packed.masks);
 masks.resize(problem.src_image_ids.size());for(auto& m:masks)m.create(height,width,CV_8UC1);
 for(int r=0;r<height;++r)for(int c=0;c<width;++c){unsigned int bits=valid_mask.at<uchar>(r,c)>=128?engine.GetPixelSelectedViews(r,c):0;for(size_t i=0;i<masks.size();++i)masks[i].at<uchar>(r,c)=((bits>>i)&1)?255:0;}
 out.views.setTo(0);fine("selected_masks_unpack");
 for(size_t i=0;i<masks.size();++i){
  cv::Mat labels(height,width,CV_32S);std::vector<int> counts;
  Connect(masks[i],labels,counts);fine("connect");Label_Update(labels,counts);fine("label_update");
  std::vector<cv::Vec3b> colors(counts.size());colors[0]=cv::Vec3b(0,0,0);
  for(size_t j=1;j<counts.size();++j){
   if(counts[j]<20*(8/problem.scale_size)*(8/problem.scale_size))colors[j]=cv::Vec3b(0,0,0);
   else colors[j]=cv::Vec3b(draw()%256,draw()%256,draw()%256);
  }
  std::vector<uchar> keep_label(colors.size());for(size_t j=0;j<colors.size();++j)keep_label[j]=colors[j]==cv::Vec3b(0,0,0);
  fine("area_color_rules");
  for(int y=0;y<height;++y){int* bits=out.views.ptr<int>(y);const int* lab=labels.ptr<int>(y);const uchar* mask=valid_mask.ptr<uchar>(y);
   for(int x=0;x<width;++x)if(mask[x]>=128&&keep_label[lab[x]])bits[x]|=(unsigned int)(1<<i);
  }
  fine("selected_bits_pack");
 }
 if(report)for(const auto& item:times)report(item.first.c_str(),item.second);
 return out;
}

template<class Engine> P3Output RunP3Post(Engine& engine,const Problem& problem,const cv::Mat& mask,const std::function<int()>& draw){return FinishP3(engine,problem,mask,PackP3(engine,problem,mask,draw),draw);}
