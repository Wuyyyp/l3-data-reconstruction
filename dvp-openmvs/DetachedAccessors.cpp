#include "APD.h"
#include "DetachedHost.h"
// Keep getter calls in a separate translation unit, as in the original APD.
float4 DetachedHost::GetPlaneHypothesis(int r,int c){float4 p;std::memcpy(&p,planes.ptr(r)+c*sizeof(float4),sizeof(p));return p;}
int DetachedHost::GetWidth(){return planes.cols;}int DetachedHost::GetHeight(){return planes.rows;}
cv::Mat DetachedHost::GetPixelStates(){return states;}cv::Mat DetachedHost::GetSelectedViews(){return views;}cv::Mat DetachedHost::GetRadiusMap(){return radius;}
float DetachedHost::GetDepthMin(){return low;}float DetachedHost::GetDepthMax(){return high;}
int DetachedHost::GetPixelSelectedViews(int r,int c){return views.at<int>(r,c);}
