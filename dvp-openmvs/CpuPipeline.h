#pragma once
#include "DetachedHost.h"
#include <mutex>
#include <streambuf>
#include <set>
// Emit complete lines: CUDA main-thread logs and CPU worker logs must not interleave.
class LineBuffer:public std::streambuf{
 std::streambuf* sink;std::mutex mutex;
 static std::string& pending(){static thread_local std::string s;return s;}
 void commit(){auto& p=pending();if(p.empty())return;std::lock_guard<std::mutex> g(mutex);sink->sputn(p.data(),p.size());sink->pubsync();p.clear();}
 protected:int overflow(int ch)override{if(ch!=traits_type::eof()){pending().push_back((char)ch);if(ch=='\n')commit();}return traits_type::not_eof(ch);}
 std::streamsize xsputn(const char* s,std::streamsize n)override{for(std::streamsize k=0;k<n;++k)overflow(s[k]);return n;}
 int sync()override{commit();return 0;}
 public:explicit LineBuffer(std::streambuf* p):sink(p){}
};
inline bool CpuPipelineOn(){const char* p=std::getenv("DVP_CPU_PIPELINE");return p&&std::atoi(p);}
struct CpuPrepared{std::shared_ptr<APD> engine;std::shared_ptr<PreparedInput> data;long long start,end;};
inline CpuPrepared PrepareCpu(const Problem& p,int blocked,std::map<int,int> versions){
 const auto start=PipeNow();auto ready=PrepareStaticInput(p);ready->versions=versions;
 auto a=std::make_shared<APD>(p);a->SetPreparedInput(ready);a->BlockSource(blocked);
 auto t=PipeNow();a->InuputInitialization();auto t2=PipeNow();a->SupportInitialization();auto end=PipeNow();
 std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=cpu_input ms="<<(t2-t)/1e6<<std::endl;
 std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=cpu_support ms="<<(end-t2)/1e6<<std::endl;
 std::cout<<"CPU_PREP ref="<<p.ref_image_id<<" start_ns="<<start<<" end_ns="<<end<<" static_ns="<<ready->end_ns-ready->start_ns<<" blocked="<<blocked<<std::endl;
 return {a,ready,start,end};
}
inline void WritePost(const Problem& p,const P3Output& o,float low,float high){
 if(!WriteBinMat(p.result_folder/path("APD_normals.dmb"),o.normal)||!WriteBinMat(p.result_folder/path("weak.bin"),o.weak)||!WriteBinMat(p.result_folder/path("selected_views.bin"),o.views))throw std::runtime_error("Post state write failed");
 if(p.params.use_radius&&!WriteBinMat(p.result_folder/path("radius.bin"),o.radius))throw std::runtime_error("Radius write failed");
 if(p.show_medium_result){ShowDepthMap(p.result_folder/path("depth_"+std::to_string(p.iteration)+".jpg"),o.depth,low,high);ShowNormalMap(p.result_folder/path("normal_"+std::to_string(p.iteration)+".jpg"),o.normal);ShowWeakImage(p.result_folder/path("weak_"+std::to_string(p.iteration)+".jpg"),o.weak);}
}
inline void ProcessPost(DetachedHost host,const Problem& p,cv::Mat mask,std::shared_ptr<std::promise<long long>> depth_ready){
 bool signaled=false;try{
 auto start=PipeNow();auto pack=PackP3(host,p,mask,[](){return std::rand();});auto packed=PipeNow();
 if(!WriteBinMat(p.result_folder/path("depths.dmb"),pack.out.depth))throw std::runtime_error("Depth publication failed");
 auto pub=PipeNow();depth_ready->set_value(pub);signaled=true;
 auto o=FinishP3(host,p,mask,std::move(pack),[](){return std::rand();},[&](const char* key,double seconds){std::cout<<"HOST_DETAIL image="<<p.ref_image_id<<" stage="<<key<<" ms="<<seconds*1000<<std::endl;});auto selected=PipeNow();
 WritePost(p,o,host.GetDepthMin(),host.GetDepthMax());auto end=PipeNow();
 std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=pixel_pack ms="<<(packed-start)/1e6<<std::endl;
 std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=depth_publish ms="<<(pub-packed)/1e6<<std::endl;
 std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=ccl_selected_pack ms="<<(selected-pub)/1e6<<std::endl;
 std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=state_write_visualization ms="<<(end-selected)/1e6<<std::endl;
 std::cout<<"CPU_POST ref="<<p.ref_image_id<<" start_ns="<<start<<" packed_ns="<<packed<<" publish_ns="<<pub<<" selected_ns="<<selected<<" end_ns="<<end<<std::endl;
 }catch(...){if(!signaled)depth_ready->set_exception(std::current_exception());throw;}
}
inline void RunCpuPipeline(const std::vector<Problem>& problems){
 const bool overlap=CpuPipelineOn();std::future<CpuPrepared> preparation;std::future<void> previous_post;
 std::future<long long> depth_ready;std::map<int,int> published;
 std::set<int> unique;for(const auto& p:problems)if(!unique.insert(p.ref_image_id).second)throw std::runtime_error("Duplicate reference ID");
 for(size_t i=0;i<problems.size();++i){
  const auto p=problems[i];const auto acquire=PipeNow();
  CpuPrepared prepared=preparation.valid()?preparation.get():PrepareCpu(p,-1,published);
  const auto acquired=PipeNow();
  auto gate_start=PipeNow();
  if(depth_ready.valid()){auto pub=depth_ready.get();published[problems[i-1].ref_image_id]=1;std::cout<<"DEPTH_PUBLISH ref="<<problems[i-1].ref_image_id<<" time_ns="<<pub<<std::endl;}
  prepared.engine->FinishDependency(i?1:0);const auto gate_end=PipeNow();
  std::cout<<"CPU_WAIT ref="<<p.ref_image_id<<" prep_ns="<<acquired-acquire<<" depth_gate_ns="<<gate_end-gate_start<<std::endl;
  auto a=prepared.engine;auto t=PipeNow();a->CudaSpaceInitialization();a->SetDataPassHelperInCuda();auto upload_end=PipeNow();
  std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=allocation_upload ms="<<(upload_end-t)/1e6<<std::endl;
  if(overlap&&i+1<problems.size()){
   const auto next=problems[i+1];const auto versions=published;int blocked=p.ref_image_id;
   preparation=std::async(std::launch::async,[next,blocked,versions](){return PrepareCpu(next,blocked,versions);});
  }
  auto gs=PipeNow();a->RunPatchMatch();auto ge=PipeNow();
  std::cout<<"GPU_SPAN ref="<<p.ref_image_id<<" start_ns="<<gs<<" end_ns="<<ge<<std::endl;
  std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=gpu_pipeline ms="<<(ge-gs)/1e6<<std::endl;
  auto snap_start=PipeNow();DetachedHost host;a->DetachOutput(host);prepared.engine.reset();a.reset();auto snap_end=PipeNow();
  std::cout<<"HOST_TIME image="<<p.ref_image_id<<" stage=snapshot_release ms="<<(snap_end-snap_start)/1e6<<std::endl;
  // Bound the post queue to one image; retain the original random draw order.
  auto drain=PipeNow();if(previous_post.valid())previous_post.get();auto drained=PipeNow();
  std::cout<<"POST_QUEUE_WAIT ref="<<p.ref_image_id<<" ns="<<drained-drain<<std::endl;
  auto promise=std::make_shared<std::promise<long long>>();depth_ready=promise->get_future();
  const cv::Mat mask=prepared.data->mask;
  if(overlap)previous_post=std::async(std::launch::async,[host,p,mask,promise](){ProcessPost(host,p,mask,promise);});
  else {ProcessPost(host,p,mask,promise);auto pub=depth_ready.get();published[p.ref_image_id]=1;std::cout<<"DEPTH_PUBLISH ref="<<p.ref_image_id<<" time_ns="<<pub<<std::endl;}
 }
 if(depth_ready.valid()){auto pub=depth_ready.get();std::cout<<"DEPTH_PUBLISH ref="<<problems.back().ref_image_id<<" time_ns="<<pub<<std::endl;}
 if(previous_post.valid())previous_post.get();
}
