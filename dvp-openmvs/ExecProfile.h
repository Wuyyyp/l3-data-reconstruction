#pragma once
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
struct ExecGpuProfile{
 struct Item{std::string label;cudaEvent_t a,b;};std::vector<Item> items;
 void begin(const std::string& s){Item q;q.label=s;cudaEventCreate(&q.a);cudaEventCreate(&q.b);cudaEventRecord(q.a);items.push_back(q);}
 void end(){cudaEventRecord(items.back().b);}
 void sync(int id){auto t=std::chrono::steady_clock::now();cudaDeviceSynchronize();std::cout<<"SYNC_TIME image="<<id<<" stage="<<items.back().label<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()<<std::endl;}
 void finish(int id){cudaDeviceSynchronize();for(auto& q:items){float ms;cudaEventElapsedTime(&ms,q.a,q.b);std::cout<<"GPU_TIME image="<<id<<" stage="<<q.label<<" ms="<<ms<<std::endl;cudaEventDestroy(q.a);cudaEventDestroy(q.b);}}
};
