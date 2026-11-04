#define main replay_entry
#include "CpuReplay.cpp"
#undef main
#include <limits>
int main(){
 try{
 for(int test=0;test<13;++test){
  int h=13+test*3,w=17+test*2;Problem p;p.scale_size=test%2?1:2;p.src_image_ids.resize(20);
  ReplayAPD initial;initial.low=1;initial.high=80;
  initial.planes=cv::Mat(h,w,CV_32FC4);initial.states=cv::Mat(h,w,CV_8UC1);initial.views=cv::Mat(h,w,CV_32SC1);initial.radius=cv::Mat(h,w,CV_32FC1);cv::Mat mask(h,w,CV_8UC1);
  std::srand(912+test);
  for(int y=0;y<h;++y)for(int x=0;x<w;++x){
   float d=(x%7==0)?0:(x%11==0)?100:10; if(test==7&&x%5==0)d=std::numeric_limits<float>::quiet_NaN();if(test==8&&x%5==0)d=std::numeric_limits<float>::infinity();
   initial.planes.at<cv::Vec4f>(y,x)=cv::Vec4f(.1f,.2f,.3f,d);initial.states.at<uchar>(y,x)=(x+y)%3;
   initial.views.at<int>(y,x)=test==0||test==12?0:test==1?((1<<20)-1):test==2?(((x+y)%2)?((1<<20)-1):0):std::rand()&((1<<20)-1);
   initial.radius.at<float>(y,x)=x%9;mask.at<uchar>(y,x)=test==3?0:((test>=4&&(x==0||y==0||x==w-1||y==h-1||(x%9==0)))?0:255);
  }
  auto fresh=[&](){ReplayAPD a=initial;a.states=initial.states.clone();a.views=initial.views.clone();a.radius=initial.radius.clone();return a;};
  force_black_test=(test==12);random_tape.clear();record_tape=true;auto a=fresh();auto baseline=Process(a,p,mask,false,false);record_tape=false;
  for(int mode=1;mode<=3;++mode){auto b=fresh();auto candidate=Process(b,p,mask,mode!=2,mode!=1);
   if(!SameSnapMat(baseline.depth,candidate.depth)||!SameSnapMat(baseline.normal,candidate.normal)||!SameSnapMat(baseline.weak,candidate.weak)||!SameSnapMat(baseline.views,candidate.views)||!SameSnapMat(baseline.radius,candidate.radius))throw std::runtime_error("selftest mismatch");
  }
  std::cout<<"SELFTEST_PASS case="<<test<<" draws="<<random_tape.size()<<std::endl;
 }
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
