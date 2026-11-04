#include "APD.h"
#include "InputOverlap.h"
#include "SnapshotIO.h"
#include <sstream>
int main(int argc,char** argv){
 try{
 if(argc!=2)return 2;std::ifstream f(argv[1]);std::string line;int count=0;
 while(std::getline(f,line)){
  std::istringstream row(line);std::string scene;int ref,n,scale,state;row>>scene>>ref>>n>>scale>>state;Problem p;p.dense_folder=path(scene);p.ref_image_id=ref;p.result_folder=p.dense_folder/path("APD")/path(ToFormatIndex(ref));p.scale_size=scale;p.params.use_edge=true;p.params.use_limit=true;p.params.use_APD=state!=0;p.params.geom_consistency=false;p.params.state=state==0?FIRST_INIT:REFINE_INIT;
  for(int i=0;i<n;++i){int id;row>>id;p.src_image_ids.push_back(id);}
  auto a=PrepareStaticInput(p);auto future=std::async(std::launch::async,[p](){return PrepareStaticInput(p);});auto b=future.get();
  if(a->width!=b->width||a->height!=b->height||a->images.size()!=b->images.size()||a->depth_min!=b->depth_min||a->depth_max!=b->depth_max)throw std::runtime_error("static metadata mismatch");
  for(size_t i=0;i<a->images.size();++i){if(!SameSnapMat(a->images[i],b->images[i]))throw std::runtime_error("static image mismatch");Camera x=a->cameras[i],y=b->cameras[i];if(std::memcmp(x.K,y.K,sizeof(x.K))||std::memcmp(x.R,y.R,sizeof(x.R))||std::memcmp(x.t,y.t,sizeof(x.t))||std::memcmp(x.c,y.c,sizeof(x.c))||x.width!=y.width||x.height!=y.height||x.depth_min!=y.depth_min||x.depth_max!=y.depth_max)throw std::runtime_error("camera mismatch");}
  if(!SameSnapMat(a->edge,b->edge)||!SameSnapMat(a->mask,b->mask))throw std::runtime_error("edge/mask mismatch");
  a->versions.clear();b->versions.clear();
  APD first(p);first.SetPreparedInput(a);first.InuputInitialization();first.SupportInitialization();first.FinishDependency(0);
  auto future_cpu=std::async(std::launch::async,[p,b](){auto e=std::make_shared<APD>(p);e->SetPreparedInput(b);e->BlockSource(p.src_image_ids.back());e->InuputInitialization();e->SupportInitialization();return e;});
  auto second=future_cpu.get();second->FinishDependency(0);
  APD direct(p);direct.InuputInitialization();direct.SupportInitialization();direct.FinishDependency(0);
  if(!first.SameCpuInput(direct))throw std::runtime_error("Serial static preparation fallback mismatch");
  if(!first.SameCpuInput(*second))throw std::runtime_error("CPU/deferred source upload data mismatch");
  std::cout<<"PREP_AUDIT_PASS ref="<<ref<<" images="<<p.src_image_ids.size()+1<<" bytes="<<a->bytes<<std::endl;++count;
 }
 std::cout<<"PREP_AUDIT_DONE images="<<count<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 3;}
}
