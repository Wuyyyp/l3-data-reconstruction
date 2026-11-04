#pragma once
#include <map>
#include <deque>
#include <mutex>
#include <future>
#include <cstdlib>
inline bool ExecOpt(){return std::getenv("DVP_EXEC_OPT") && std::atoi(std::getenv("DVP_EXEC_OPT"));}
inline cv::Mat ExecRead(const std::string& name,int flags){
 if(!ExecOpt())return cv::imread(name,flags);
 static std::mutex mu; static std::map<std::string,cv::Mat> cache; static std::deque<std::string> fifo; static size_t bytes=0;
 const std::string key=name+":"+std::to_string(flags);
 {std::lock_guard<std::mutex> lock(mu);auto it=cache.find(key);if(it!=cache.end())return it->second.clone();}
 cv::Mat value=cv::imread(name,flags);if(value.empty())return value;
 std::lock_guard<std::mutex> lock(mu);
 if(!cache.count(key)){while(bytes+value.total()*value.elemSize()>1024ULL*1024*1024 && !fifo.empty()){auto it=cache.find(fifo.front());bytes-=it->second.total()*it->second.elemSize();cache.erase(it);fifo.pop_front();}cache[key]=value;fifo.push_back(key);bytes+=value.total()*value.elemSize();}
 return value.clone();
}
struct ExecDevicePool{std::map<void*,size_t> active;std::multimap<size_t,void*> idle;size_t bytes=0;};
inline ExecDevicePool& ExecPool(){static ExecDevicePool p;return p;}
inline cudaError_t ExecMalloc(void** ptr,size_t n){
 if(!ExecOpt())return cudaMalloc(ptr,n);auto& p=ExecPool();auto it=p.idle.lower_bound(n);
 if(it!=p.idle.end() && it->first<=n*2){*ptr=it->second;p.active[*ptr]=it->first;p.bytes-=it->first;p.idle.erase(it);return cudaSuccess;}
 auto e=cudaMalloc(ptr,n);if(e==cudaSuccess)p.active[*ptr]=n;return e;
}
inline cudaError_t ExecFree(void* ptr){
 if(!ExecOpt())return cudaFree(ptr);auto& p=ExecPool();auto it=p.active.find(ptr);
 if(it==p.active.end())return cudaFree(ptr);size_t n=it->second;p.active.erase(it);
 if(p.bytes+n>2ULL*1024*1024*1024)return cudaFree(ptr);p.idle.emplace(n,ptr);p.bytes+=n;return cudaSuccess;
}
