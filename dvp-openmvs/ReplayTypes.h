#pragma once
#include "SnapshotIO.h"
struct float4{float x,y,z,w;};
enum{WEAK,STRONG,UNKNOWN};
struct Problem{std::vector<int> src_image_ids;int scale_size;};
struct ReplayAPD{
 cv::Mat planes,states,views,radius;float low,high;
 float4 GetPlaneHypothesis(int r,int c){auto p=planes.at<cv::Vec4f>(r,c);return {p[0],p[1],p[2],p[3]};}
 int GetWidth(){return planes.cols;}int GetHeight(){return planes.rows;}
 cv::Mat GetPixelStates(){return states;}cv::Mat GetSelectedViews(){return views;}cv::Mat GetRadiusMap(){return radius;}
 float GetDepthMin();float GetDepthMax();
 int GetPixelSelectedViews(int r,int c){return views.at<int>(r,c);}
 void SetPixelSelectedViews(int r,int c,int v){views.at<int>(r,c)=v;}
};
