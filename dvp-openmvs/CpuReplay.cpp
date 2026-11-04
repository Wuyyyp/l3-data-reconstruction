#include "SnapshotIO.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <map>
#include "ReplayTypes.h"
bool force_black_test=false;
std::vector<int> random_tape;size_t random_cursor=0;bool record_tape=false;
int ReplayRand(){if(record_tape){random_tape.push_back(force_black_test?0:std::rand());return random_tape[random_cursor++];}if(random_cursor>=random_tape.size())throw std::runtime_error("random tape exhausted");return random_tape[random_cursor++];}
void setBit_YZL(unsigned int* input,const unsigned int n){(*input)|=(unsigned int)(1<<n);}
#include "ReplayCCL.h"
#include "P3Post.h"
struct Result{cv::Mat depth,normal,weak,views,radius;std::vector<std::pair<std::string,double>> stages;};
Result Process(ReplayAPD& APD,const Problem& problem,const cv::Mat& valid_mask,bool p3,bool p4){
 int width=APD.planes.cols,height=APD.planes.rows;
 cv::Mat depth(height,width,CV_32FC1),normal(height,width,CV_32FC3),pixel_states=APD.states;
 Result result;auto clock=std::chrono::steady_clock::now();
 auto stamp=[&](const char* name){auto now=std::chrono::steady_clock::now();result.stages.push_back({name,std::chrono::duration<double>(now-clock).count()});clock=now;};
 random_cursor=0;
 std::map<std::string,double> detail;
 auto fine_clock=std::chrono::steady_clock::now();
 auto fine=[&](const char* name){auto now=std::chrono::steady_clock::now();detail[name]+=std::chrono::duration<double>(now-fine_clock).count();fine_clock=now;};
 if(!p3){
	//yzl
	unsigned int views;
	std::vector<cv::Mat> matVector;
	std::vector<cv::Mat> matVector_uchar;
	for (int i = 0; i < problem.src_image_ids.size(); ++i) {
		cv::Mat_<cv::Vec3b> tempImage(height, width, CV_8UC3);
		cv::Mat_<uchar> tempImage_uchar(height, width);
		matVector.push_back(tempImage);
		matVector_uchar.push_back(tempImage_uchar);
	}

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			float4 plane_hypothesis = APD.GetPlaneHypothesis(r, c);
			const bool keep = valid_mask.at<uchar>(r, c) >= 128;
			depth.at<float>(r, c) = keep ? plane_hypothesis.w : 0.0f;
			if (!keep || depth.at<float>(r, c) < APD.GetDepthMin() || depth.at<float>(r, c) > APD.GetDepthMax()) {
				depth.at<float>(r, c) = 0;
				pixel_states.at<uchar>(r, c) = UNKNOWN;
			}
			normal.at<cv::Vec3f>(r, c) = keep
				? cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z)
				: cv::Vec3f(0.0f, 0.0f, 0.0f);

			views = keep ? APD.GetPixelSelectedViews(r, c) : 0;
			for (int i = 0; i < problem.src_image_ids.size(); ++i) {
				if ((views >> i) & 1) {
					matVector[i].at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
					matVector_uchar[i].at<uchar>(r, c) = 255;

				}
				else {
					matVector[i].at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
					matVector_uchar[i].at<uchar>(r, c) = 0;
				}
			}
		}
	}

 stamp("pixel_pack_and_masks");
	cv::Mat reusable_labels; std::vector<int> reusable_counts;
	for (int i = 0; i < problem.src_image_ids.size(); ++i) {
		cv::Mat lab_mask; if(p4){reusable_labels.create(height,width,CV_32S);lab_mask=reusable_labels;}else lab_mask=cv::Mat(height,width,CV_32S);
		std::vector<int> own_counts; std::vector<int>& label_cnt=p4?reusable_counts:own_counts;label_cnt.clear();

		fine_clock=std::chrono::steady_clock::now();
		if(p4)ConnectP4(matVector_uchar[i], lab_mask, label_cnt);else Connect(matVector_uchar[i], lab_mask, label_cnt);fine("connect");
		Label_Update(lab_mask, label_cnt);fine("label_update");

		int label_num = label_cnt.size();
		std::vector<cv::Vec3b> colors(label_num);
		colors[0] = cv::Vec3b(0, 0, 0);
		for (int j = 1; j < label_num; j++) {
			if (label_cnt[j] < 20 * (8 / problem.scale_size) * (8 / problem.scale_size))
				colors[j] = cv::Vec3b(0, 0, 0);
			else
				colors[j] = cv::Vec3b(ReplayRand() % 256, ReplayRand() % 256, ReplayRand() % 256);
		}
		fine("area_color_rules");
		cv::Mat img_connect(height, width, CV_8UC3);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int label = lab_mask.at<int>(y, x);
				//matVector[i].at<cv::Vec3b>(y, x) = colors[label];
				if (colors[label] != cv::Vec3b(0, 0, 0)) {
					matVector[i].at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				}
				else {
					matVector[i].at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
				}
			}
		}
		fine("labels_to_rgb");
	}

 stamp("connected_components_and_labels");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (valid_mask.at<uchar>(y, x) < 128) {
				APD.SetPixelSelectedViews(y, x, 0);
				continue;
			}
			unsigned int temp_selected_views = 0;
			for (int i = 0; i < problem.src_image_ids.size(); ++i) {
				if (matVector[i].at<cv::Vec3b>(y, x) == cv::Vec3b(255, 255, 255))
					setBit_YZL(&temp_selected_views, i);
			}
			APD.SetPixelSelectedViews(y, x, temp_selected_views);
		}
	}

	stamp("selected_views_repack");
 }else{
 auto output=RunP3Post(APD,problem,valid_mask,[](){return ReplayRand();});
 depth=output.depth;normal=output.normal;pixel_states=output.weak;
 stamp("integrated_p3_post");

 }
 if(random_cursor!=random_tape.size())throw std::runtime_error("random draw count differs");
 for(auto& item:detail)result.stages.push_back({"detail_"+item.first,item.second});
 result.depth=depth;result.normal=normal;result.weak=pixel_states;result.views=APD.views;result.radius=APD.radius;return result;
}
int main(int argc,char** argv){
 try{
 if(argc!=4)return 2;std::string folder=argv[1],mode=argv[2],prefix=argv[3];
 cv::setNumThreads(1);
 ReplayAPD a;a.planes=LoadSnapMat(folder+"/planes.snap");a.states=LoadSnapMat(folder+"/weak.snap");a.views=LoadSnapMat(folder+"/views.snap");a.radius=LoadSnapMat(folder+"/radius.snap");
 cv::Mat mask=LoadSnapMat(folder+"/mask.snap");Problem problem;int n;std::ifstream meta(folder+"/meta.txt");meta>>a.low>>a.high>>n>>problem.scale_size;problem.src_image_ids.resize(n);
 std::ifstream tape(folder+"/random.bin",std::ios::binary|std::ios::ate);size_t bytes=tape.tellg();random_tape.resize(bytes/sizeof(int));tape.seekg(0);tape.read((char*)random_tape.data(),bytes);
 auto before=std::chrono::steady_clock::now();Result r=Process(a,problem,mask,mode=="p3"||mode=="p3p4",mode=="p4"||mode=="p3p4");double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
 std::vector<std::pair<std::string,cv::Mat>> mats={{"depth",r.depth},{"normal",r.normal},{"weak",r.weak},{"views",r.views},{"radius",r.radius}};
 bool same=true;for(auto& m:mats){bool exact=SameSnapMat(m.second,LoadSnapMat(folder+"/expected_"+m.first+".snap"));same&=exact;std::cout<<"EXACT name="<<m.first<<" value="<<exact<<"\n";if(prefix!="-")SaveSnapMat(prefix+"_"+m.first+".snap",m.second);}
 for(auto& s:r.stages)std::cout<<"CPU_STAGE name="<<s.first<<" seconds="<<std::setprecision(12)<<s.second<<"\n";
 std::cout<<"CPU_REPLAY_DONE mode="<<mode<<" seconds="<<std::setprecision(12)<<elapsed<<" draws="<<random_cursor<<" exact="<<same<<"\n";return same?0:5;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 3;}
}
