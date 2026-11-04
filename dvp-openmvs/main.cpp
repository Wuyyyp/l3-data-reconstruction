#include "APD.h"
#include "StaticInputCache.h"
#include "InputOverlap.h"
#include "P3Post.h"
#include "CpuPipeline.h"
#include "main.h"
#include "APD.h"
#include "IterationSchedule.h"
#include <thread>
#include "ExecHelpers.h"

using namespace boost::filesystem;

void setBit_YZL(unsigned int* input, const unsigned int n)
{
	(*input) |= (unsigned int)(1 << n);
}

// 求连通区域
void Connect_RGB(const cv::Mat& dst_image, cv::Mat& label_mask, std::vector<int>& label_cnt) {
	std::vector<std::vector<int>> left_neigh(dst_image.rows);
	std::vector<std::vector<int>> up_neigh(dst_image.rows);

	for (int y = 0; y < dst_image.rows; y++) {
		left_neigh[y].resize(dst_image.cols);
		up_neigh[y].resize(dst_image.cols);
		for (int x = 0; x < dst_image.cols; x++) {
			// 左连通
			if (x == 0) {
				left_neigh[y][x] = 0;
			}
			else {
				if (dst_image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0) && dst_image.at<cv::Vec3b>(y, x - 1) == cv::Vec3b(0, 0, 0)) {
					left_neigh[y][x] = 1;
				}
				else {
					left_neigh[y][x] = 0;
				}
			}
			// 上连通
			if (y == 0) {
				up_neigh[y][x] = 0;
			}
			else {
				if (dst_image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0) && dst_image.at<cv::Vec3b>(y - 1, x) == cv::Vec3b(0, 0, 0)) {
					up_neigh[y][x] = 1;
				}
				else {
					up_neigh[y][x] = 0;
				}
			}
		}
	}

	// 维护一个并查集
	int cnt = 1;
	std::vector<int> connection;
	connection.push_back(0);
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			if (dst_image.at<uchar>(y, x) == 255) {
				label_mask.at<int>(y, x) = 0;
			}
			else {
				bool left = false, up = false;
				if (left_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					left = true;
				}
				if (up_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					up = true;
				}
				if (left == false && up == false) {
					label_mask.at<int>(y, x) = cnt;
					connection.push_back(cnt);
					cnt++;
				}
				else if (left == true && up == true) {
					int left_label = label_mask.at<int>(y, x - 1);
					int up_label = label_mask.at<int>(y - 1, x);
					if (left_label > up_label) {
						connection[left_label] = up_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					}
					else if (left_label < up_label) {
						connection[up_label] = left_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					}
				}
			}
		}
	}

	for (size_t i = 1; i < connection.size(); i++) {
		int cur_label = connection[i];
		int pre_label = connection[cur_label];
		while (pre_label != cur_label) {
			cur_label = pre_label;
			pre_label = connection[pre_label];
		}
		connection[i] = cur_label;
	}

	int label_num = 1;
	std::vector<int> mapping;
	mapping.push_back(0);
	for (size_t i = 1; i < connection.size(); i++) {
		mapping.push_back(0);
		if (connection[i] == (int)i) {
			mapping[i] = label_num;
			label_num++;  //标签总数
		}
	}

	// 重编号
	for (size_t i = 1; i < connection.size(); i++) {
		connection[i] = mapping[connection[i]];
	}

	for (int i = 0; i < label_num; i++) {
		label_cnt.push_back(0);
	}

	// 连通区域计数
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			int label = label_mask.at<int>(y, x);
			label_mask.at<int>(y, x) = connection[label];
			label_cnt[connection[label]]++;
		}
	}
}

void GenerateSampleList(const path& dense_folder, std::vector<Problem>& problems)
{
	path cluster_list_path = dense_folder / path("pair.txt");
	problems.clear();
	ifstream file(cluster_list_path);
	std::stringstream iss;
	std::string line;

	int num_images = 0;
	if (!file) throw std::runtime_error("Cannot open pair.txt");
	iss.clear();
	std::getline(file, line);
	iss.str(line);
	if (!(iss >> num_images) || num_images <= 0) throw std::runtime_error("Invalid pair.txt image count");

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.index = i;
		problem.src_image_ids.clear();
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> problem.ref_image_id;

		problem.dense_folder = dense_folder;
		problem.result_folder = dense_folder / path("APD") / path(ToFormatIndex(problem.ref_image_id));
		create_directory(problem.result_folder);

		int num_src_images = 0;
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		if (!(iss >> num_src_images) || num_src_images < 1 || num_src_images >= MAX_IMAGES) throw std::runtime_error("pair.txt requires 1..31 source views");
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			if (!(iss >> id >> score)) throw std::runtime_error("Malformed pair.txt source view");
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		if (problem.src_image_ids.empty()) throw std::runtime_error("No positive-score source views");
		problems.push_back(problem);
	}
}

bool CheckImages(const std::vector<Problem>& problems) {
	if (problems.size() == 0) {
		return false;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return false;
	}
	const int width = image.cols;
	const int height = image.rows;
	for (size_t i = 1; i < problems.size(); ++i) {
		image_path = problems[i].dense_folder / path("images") / path(ToFormatIndex(problems[i].ref_image_id) + ".jpg");
		image = cv::imread(image_path.string());
		if (image.cols != width || image.rows != height) {
			return false;
		}
	}
	return true;
}

void GetProblemEdges(const Problem& problem) {
	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	int scale = 0;
	while ((1 << scale) < problem.scale_size) scale++;

	path image_folder = problem.dense_folder / path("images");
	path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
	cv::Mat image_uint = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
	cv::Mat src_img;
	image_uint.convertTo(src_img, CV_32FC1);
	const float factor = 1.0f / (float)(problem.scale_size);
	const int new_cols = std::round(src_img.cols * factor);
	const int new_rows = std::round(src_img.rows * factor);
	cv::Mat scaled_image_float;
	cv::resize(src_img, scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
	scaled_image_float.convertTo(src_img, CV_8UC1);

	if (problem.params.use_edge) {
		// path edge_path = problem.result_folder / path("edges.dmb");
		path edge_path = problem.result_folder / path("edges_" + std::to_string(scale) + ".dmb");
		std::ifstream edge_file(edge_path.string());
		bool edge_exists = edge_file.good();
		edge_file.close();
		if (!edge_exists) {
			cv::Mat edge = EdgeSegment(scale, src_img, 0, true);
			WriteBinMat(edge_path, edge);
			if (problem.show_medium_result) {
				path ref_image_edge_path = problem.result_folder / path("rawedge_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_edge_path.string(), edge);
			}
		}
	}

	if (problem.params.use_label) {
		// path label_path = problem.result_folder / path("labels.dmb");
		path label_path = problem.result_folder / path("labels_" + std::to_string(scale) + ".dmb");
		std::ifstream label_file(label_path.string());
		bool label_exists = label_file.good();
		label_file.close();
		if (!label_exists) {
			cv::Mat label = EdgeSegment(scale, image_uint, 1);
			WriteBinMat(label_path, label);
			if (problem.show_medium_result) {
				path ref_image_con_path = problem.result_folder / path("connect_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_con_path.string(), EdgeSegment(scale, image_uint, -1));
			}
		}
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int ComputeRoundNum(const std::vector<Problem>& problems) {
	if (problems.size() == 0) {
		return 0;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return 0;
	}
	int max_size = MAX(image.cols, image.rows);
	int round_num = 1;
	while (max_size > 800) {
		max_size /= 2;
		round_num++;
	}
	return round_num;
}


void ProcessProblem(const Problem& problem) {
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	std::cout << "Iteration: " << problem.iteration << std::endl;

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	APD APD(problem);
	float depth_min = APD.GetDepthMin();
	float depth_max = APD.GetDepthMax();
	APD.InuputInitialization();
	APD.SupportInitialization();
	APD.CudaSpaceInitialization();
	APD.SetDataPassHelperInCuda();
	APD.RunPatchMatch();

	int width = APD.GetWidth(), height = APD.GetHeight();
	cv::Mat depth = cv::Mat(height, width, CV_32FC1);
	cv::Mat normal = cv::Mat(height, width, CV_32FC3);
	cv::Mat pixel_states = APD.GetPixelStates();
	cv::Mat valid_mask(height, width, CV_8UC1, cv::Scalar(255));
	path block_path = problem.dense_folder / path("blocks") / path("mask_" + std::to_string(problem.ref_image_id) + ".png");
	if (!exists(block_path)) block_path.replace_extension(".jpg");
	if (exists(block_path)) {
		cv::Mat block = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
		if (block.empty()) throw std::runtime_error("Cannot read dense matching mask: " + block_path.string());
		cv::Mat ignored, scaled;
		cv::compare(block, 128, ignored, cv::CMP_LT);
		ignored.convertTo(ignored, CV_32F, 1.0 / 255.0);
		cv::resize(ignored, scaled, cv::Size(width, height), 0, 0, cv::INTER_AREA);
		cv::compare(scaled, 0, valid_mask, cv::CMP_EQ);
	}

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

	for (int i = 0; i < problem.src_image_ids.size(); ++i) {
		cv::Mat lab_mask(height, width, CV_32S);
		std::vector<int> label_cnt;

		Connect(matVector_uchar[i], lab_mask, label_cnt);
		Label_Update(lab_mask, label_cnt);

		int label_num = label_cnt.size();
		std::vector<cv::Vec3b> colors(label_num);
		colors[0] = cv::Vec3b(0, 0, 0);
		for (int j = 1; j < label_num; j++) {
			if (label_cnt[j] < 20 * (8 / problem.scale_size) * (8 / problem.scale_size))
				colors[j] = cv::Vec3b(0, 0, 0);
			else
				colors[j] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
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
	}

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

	path depth_path = problem.result_folder / path(problem.defer_depth_publish ? "depths.next.dmb" : "depths.dmb");
	if (!WriteBinMat(depth_path, depth)) throw std::runtime_error("Failed to write DVP depth map");
	path normal_path = problem.result_folder / path("APD_normals.dmb");
	WriteBinMat(normal_path, normal);
	path weak_path = problem.result_folder / path("weak.bin");
	WriteBinMat(weak_path, pixel_states);
	path selected_view_path = problem.result_folder / path("selected_views.bin");
	WriteBinMat(selected_view_path, APD.GetSelectedViews());
	if (problem.params.use_radius) {
		path radius_path = problem.result_folder / path("radius.bin");
		WriteBinMat(radius_path, APD.GetRadiusMap());
	}

	if (problem.iteration == 15) {
		path depths_geom = problem.result_folder / path("depths_geom.dmb");
		writeDepthDmb(depths_geom, depth);
		path normals_path = problem.result_folder / path("normals.dmb");
		writeNormalDmb(normals_path, normal);
		path weak_img_path_2 = problem.result_folder / path("weak.png");
		ShowWeakImage(weak_img_path_2, pixel_states);
	}


	//yzl
	//for (int i = 0; i < problem.src_image_ids.size(); ++i) {
	//	std::string view_name = "view_" + std::to_string(i) + ".png";
	//	path view_path = problem.result_folder / path(view_name);
	//	/*std::cout << "view_path.string(): " << view_path.string() << std::endl;*/
	//	cv::imwrite(view_path.string(), matVector[i]);
	//}

	if (problem.show_medium_result) {
		path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
		path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
		path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".jpg");
		ShowDepthMap(depth_img_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
		ShowNormalMap(normal_img_path, normal);
		ShowWeakImage(weak_img_path, pixel_states);

		// if ((problem.iteration + 1) % 4 == 0) {
		// 	path image_folder = problem.dense_folder / path("images");
		// 	path cam_folder = problem.dense_folder / path("cams");
		// 	path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		// 	path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		// 	path point_cloud_path = problem.result_folder / path("point_" + std::to_string(problem.iteration) + ".ply");
		// 	// path point_cloud_path = problem.result_folder / path("point_test_" + std::to_string(problem.iteration) + ".ply");

		// 	// for (int r = 0; r < height; ++r) for (int c = 0; c < width; ++c) if (pixel_states.at<uchar>(r, c) != STRONG) depth.at<float>(r, c) = 0;
		// 	ExportDepthImagePointCloud(point_cloud_path, image_path, cam_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
		// }
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int main(int argc, char** argv) {
 LineBuffer lines(std::cout.rdbuf());
 struct Restore{std::streambuf* p;~Restore(){std::cout.rdbuf(p);}}restore{std::cout.rdbuf(&lines)};
 const char* cpu_seed = std::getenv("DVP_TEST_SEED");
 if (cpu_seed) std::srand(static_cast<unsigned int>(std::strtoull(cpu_seed, nullptr, 10)));
	if (argc < 2) {
		std::cerr << "USAGE: APD dense_folder [gpu_index [worker_index worker_count]]\n";
		return EXIT_FAILURE;
	}
	path dense_folder(argv[1]);
	path output_folder = dense_folder / path("APD");
	create_directory(output_folder);
	// set cuda device for multi-gpu machine
	int gpu_index = 0;
	if (argc == 3) {
		gpu_index = std::atoi(argv[2]);
	}
	if (argc >= 3) gpu_index = std::atoi(argv[2]);
	int worker_index = 0;
	int worker_count = 1;
	if (argc == 5) {
		worker_index = std::atoi(argv[3]);
		worker_count = std::atoi(argv[4]);
		if (worker_count < 1 || worker_index < 0 || worker_index >= worker_count) {
			std::cerr << "Invalid worker index/count\n";
			return EXIT_FAILURE;
		}
	} else if (argc != 2 && argc != 3) {
		std::cerr << "USAGE: APD dense_folder [gpu_index [worker_index worker_count]]\n";
		return EXIT_FAILURE;
	}
	const char* profile_env = std::getenv("DVP_ITERATION_PROFILE");
	const std::string profile = profile_env ? profile_env : "legacy";
	// Validate before touching CUDA or any existing result maps.
	MakeIterationSchedule(profile == "openmvs" ? 3 : (profile == "legacy19" ? 2 : 1), profile);
 const bool cpu_pipeline = CpuPipelineOn();
 if (cpu_pipeline && (worker_count != 1 || !UsesInPlaceDepths(profile)))
     throw std::runtime_error("CPU pipeline requires a single worker and legacy or legacy19 in-place profile");
 if (cpu_pipeline && (ExecOpt() || StaticCacheOn() || InputOverlapOn()))
     throw std::runtime_error("CPU pipeline must not be combined with experimental cache/exec/input-overlap switches");
	CUDA_SAFE_CALL(cudaSetDevice(gpu_index));
	// generate problems
	std::vector<Problem> all_problems;
	GenerateSampleList(dense_folder, all_problems);
	if (!CheckImages(all_problems)) {
		std::cerr << "Missing images or inconsistent image sizes\n";
		return EXIT_FAILURE;
	}
	std::cout << "Dense matching mask mode: DVP_DENSE_MASK_V1 DVP_DENSE_MASK_V2\n";
	std::vector<Problem> problems;
	for (const Problem& problem : all_problems) {
		if (problem.index % worker_count == worker_index) problems.push_back(problem);
	}
	std::cout << "Worker " << worker_index << "/" << worker_count << " on GPU " << gpu_index
		<< " processes " << problems.size() << " of " << all_problems.size() << " problems" << std::endl;


 int round_num = profile == "openmvs" ? 3 : ComputeRoundNum(all_problems);
	if (profile == "openmvs") std::cout << "DVP_OPENMVS_SCHEDULE_V1" << std::endl;
	path barrier_folder = dense_folder / path("APD_parallel_barrier");
	if (worker_count > 1) create_directories(barrier_folder);
	auto barrier = [&](int iteration) {
		if (worker_count == 1) return;
		const std::string prefix = "iteration_" + std::to_string(iteration) + "_worker_";
		const path marker = barrier_folder / path(prefix + std::to_string(worker_index) + ".done");
		{
			std::ofstream out(marker.string());
			out << "done\n";
		}
		while (true) {
			int ready = 0;
			for (directory_iterator it(barrier_folder), end; it != end; ++it) {
				if (is_regular_file(it->path()) && it->path().filename().string().find(prefix) == 0) ++ready;
			}
			if (ready >= worker_count) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(UsesInPlaceDepths(profile) ? 1000 : 20));
		}
		std::cout << "Worker barrier completed for iteration " << iteration << std::endl;
	};

	std::cout << "Round nums: " << round_num << std::endl;
	const auto schedule = MakeIterationSchedule(round_num, profile);

	std::cout << "DVP_ITERATION_POLICY_V1 profile=" << profile
		<< " passes=" << schedule.size() << " frozen_geometry=" << (!UsesInPlaceDepths(profile)) << std::endl;
	for (size_t iteration_index = 0; iteration_index < schedule.size(); ++iteration_index) {
		const auto& pass = schedule[iteration_index];
		const int i = pass.scale_index;
		const bool geometry = pass.geometry_index >= 0;
		const bool defer_depth = geometry && !UsesInPlaceDepths(profile);
		const auto started = std::chrono::steady_clock::now();
		std::cout << "DVP_PASS_START pass=" << iteration_index << " scale=" << i
			<< " geometry=" << geometry << " inner=" << pass.inner_iterations
			<< " weak_peak=" << pass.weak_peak_radius << std::endl;
		for (auto& problem : problems) {
			problem.iteration = static_cast<int>(iteration_index);
			problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i));
			problem.defer_depth_publish = defer_depth;
			auto& params = problem.params;
			params.state = geometry ? REFINE_ITER : (i == 0 ? FIRST_INIT : REFINE_INIT);
			params.use_APD = i > 0;
			if (geometry || i > 0) {
				const int parameter_level = profile == "openmvs" ? std::max(0, i - 1) : i;
				params.ransac_threshold = 0.01 - parameter_level * 0.00125;
				params.rotate_time = MIN(static_cast<int>(std::pow(2, parameter_level)), 4);
			}
			if (!geometry && i > 0) params.use_detail = i < round_num - 1;
			params.geom_consistency = geometry;
			params.max_iterations = pass.inner_iterations;
			params.weak_peak_radius = pass.weak_peak_radius;
			if (!geometry) GetProblemEdges(problem);
            if (!cpu_pipeline) ProcessProblem(problem);
		}
 if (cpu_pipeline) RunCpuPipeline(problems);

		barrier(static_cast<int>(iteration_index));
		if (defer_depth) {
			// Every worker has finished reading the previous depth maps. Publish only
			// now, then wait again so no reader enters the next pass prematurely.
			for (const auto& problem : problems) {
				const path pending = problem.result_folder / "depths.next.dmb";
				if (!exists(pending) || file_size(pending) == 0)
					throw std::runtime_error("Missing pending DVP depth map: " + pending.string());
				rename(pending, problem.result_folder / "depths.dmb");
			}
			barrier(-static_cast<int>(iteration_index) - 1);
		}
		const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
		PrintStaticCacheStats();
		std::cout << "DVP_PASS_DONE pass=" << iteration_index << " elapsed_s=" << elapsed << std::endl;
	}

	if (worker_index == 0) {
		const char* skip_fusion = std::getenv("DVP_SKIP_FUSION");
		if (skip_fusion != nullptr && std::atoi(skip_fusion) != 0) {
			std::cout << "Skipping DVP native fusion (DVP_SKIP_FUSION is set)\n";
		}
		else {
			RunFusion(dense_folder, all_problems);
		}
	}
	// {// delete files
	// 	for (size_t i = 0; i < problems.size(); ++i) {
	// 		const auto &problem = problems[i];
	// 		remove(problem.result_folder / path("weak.bin"));
	// 		remove(problem.result_folder / path("depths.dmb"));
	// 		remove(problem.result_folder / path("APD_normals.dmb"));
	// 		remove(problem.result_folder / path("selected_views.bin"));
	// 		remove(problem.result_folder / path("neighbour.bin")); 
	// 		remove(problem.result_folder / path("neighbour_map.bin"));
	// 	}
	// }
	std::cout << "All done\n";
	return EXIT_SUCCESS;
}
