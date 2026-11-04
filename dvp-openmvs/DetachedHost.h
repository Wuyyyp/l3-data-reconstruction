#pragma once
// Owns GPU readback independently of CUDA allocations and of the next reference.
struct DetachedHost {
 cv::Mat planes,states,views,radius;float low,high;
 float4 GetPlaneHypothesis(int r,int c);
 int GetWidth();int GetHeight();
 cv::Mat GetPixelStates();cv::Mat GetSelectedViews();cv::Mat GetRadiusMap();
 float GetDepthMin();float GetDepthMax();
 int GetPixelSelectedViews(int r,int c);
};
