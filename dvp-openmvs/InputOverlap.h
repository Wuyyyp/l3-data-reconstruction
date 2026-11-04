#pragma once
#include <memory>
#include <future>
#include <chrono>
#include <map>
#include "StaticInputCache.h"
inline long long PipeNow(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline bool InputOverlapOn(){const char* x=std::getenv("DVP_INPUT_OVERLAP");return x&&std::atoi(x);}
struct PreparedInput {
 std::vector<cv::Mat> images;std::vector<Camera> cameras;
 std::map<int,int> versions;
 int width=0,height=0,num_images=0,ref=-1;float depth_min=0,depth_max=0;
 cv::Mat edge,mask;long long start_ns=0,end_ns=0;size_t bytes=0;
};
inline std::map<int,int>& PublishedInputs(){static std::map<int,int> m;return m;}
inline int PublishedInputVersion(int id){auto it=PublishedInputs().find(id);return it==PublishedInputs().end()?0:it->second;}
inline std::shared_ptr<PreparedInput> PrepareStaticInput(const Problem& problem){
 auto data=std::make_shared<PreparedInput>();data->ref=problem.ref_image_id;data->start_ns=PipeNow();
 auto& images=data->images;auto& cameras=data->cameras;auto& width=data->width;auto& height=data->height;auto& num_images=data->num_images;
	images.clear();
	cameras.clear();
	// get folder
	path image_folder = problem.dense_folder / path("images");
	path cam_folder = problem.dense_folder / path("cams");
	//path weak_folder = problem.dense_folder / path("weaks");
	// =================================================
	// read ref image and src images
	// ref
	{
		path ref_image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		cv::Mat image_float = StaticGrayFloat(ref_image_path.string());
		images.push_back(image_float);
		width = image_float.cols;
		height = image_float.rows;
	}
	// src
	for (const auto& src_idx : problem.src_image_ids) {
		path src_image_path = image_folder / path(ToFormatIndex(src_idx) + ".jpg");
		cv::Mat Resize_Image = StaticSourceFloat(src_image_path.string(), height, width);
		images.push_back(Resize_Image);
		// assert: images_float.cols == width;
		// assert: images_float.rows == height;
	}
	if (images.size() > MAX_IMAGES) {
		std::cerr << "Can't process so much images: " << images.size() << std::endl;
		exit(EXIT_FAILURE);
	}
	// =================================================
	// read ref camera and src camera
	// ref
	{
		path ref_cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera cam;
		StaticReadCamera(ref_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// src
	for (const auto& src_idx : problem.src_image_ids) {
		path src_cam_path = cam_folder / path(ToFormatIndex(src_idx) + "_cam.txt");
		Camera cam;
		StaticReadCamera(src_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// =================================================
	// set some params
	data->depth_min = cameras[0].depth_min * 0.6f;
	data->depth_max = cameras[0].depth_max * 1.2f;
	data->num_images = (int)images.size();
	num_images = (int)images.size();
	// =================================================

	// =================================================
	// scale images
	if (problem.scale_size != 1) {
		for (int i = 0; i < num_images; ++i) {
			const float factor = 1.0f / (float)(problem.scale_size);
			const int new_cols = std::round(images[i].cols * factor);
			const int new_rows = std::round(images[i].rows * factor);

			const float scale_x = new_cols / static_cast<float>(images[i].cols);
			const float scale_y = new_rows / static_cast<float>(images[i].rows);

			cv::Mat_<float> scaled_image_float;
			cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
			images[i] = scaled_image_float.clone();

			width = scaled_image_float.cols;
			height = scaled_image_float.rows;

			cameras[i].K[0] *= scale_x;
			cameras[i].K[2] *= scale_x;
			cameras[i].K[4] *= scale_y;
			cameras[i].K[5] *= scale_y;
			cameras[i].width = width;
			cameras[i].height = height;
		}
	}
	// =================================================
 int scale=0;while((1<<scale)<problem.scale_size)++scale;
 if(problem.params.use_edge||problem.params.use_limit)data->edge=StaticEdge(problem.result_folder/path("edges_"+std::to_string(scale)+".dmb"));
 data->mask=cv::Mat(height,width,CV_8UC1,cv::Scalar(255));
 path block=problem.dense_folder/path("blocks")/path("mask_"+std::to_string(problem.ref_image_id)+".png");
 if(!exists(block))block.replace_extension(".jpg");
 if(exists(block))data->mask=StaticValidMask(block,width,height);
 for(auto& m:images)data->bytes+=m.total()*m.elemSize();data->bytes+=data->edge.total()*data->edge.elemSize()+data->mask.total()*data->mask.elemSize();

 if (const char* folder = std::getenv("DVP_FAR_MASK_DIR")) {
  if(*folder) {
  cv::Mat full = cv::imread(std::string(folder)+"/mask_"+std::to_string(problem.ref_image_id)+".png",cv::IMREAD_GRAYSCALE);
  if(full.empty())throw std::runtime_error("Missing far-region mask");
  cv::Mat active,scaled;
  cv::compare(full,128,active,cv::CMP_GE);
  active.convertTo(active,CV_32F,1.0/255);
  cv::resize(active,scaled,cv::Size(width,height),0,0,cv::INTER_AREA);
  cv::compare(scaled,0,active,cv::CMP_GT); // Preserve any active support at coarse scales.
  cv::bitwise_and(data->mask,active,data->mask);
  }
 }
 data->end_ns=PipeNow();return data;
}
