void Label_Seek(int label_1, int label_2, std::vector<std::vector<int>>& connection) {
	bool find_1 = false;
	bool find_2 = false;
	int ind_1 = -1;
	int ind_2 = -1;
	for (int y = 0; y < connection.size(); y++) {
		for (int x = 0; x < connection[y].size(); x++) {
			if (find_1 && find_2)
				break;
			if (label_1 == connection[y][x]) {
				find_1 = true;
				ind_1 = y;
			}
			if (label_2 == connection[y][x]) {
				find_2 = true;
				ind_2 = y;
			}
		}
		if (find_1 && find_2)
			break;
	}
	if (!find_1 && !find_2) { //两个都没找到
		std::vector<int> New;
		New.push_back(label_1);
		New.push_back(label_2);
		connection.push_back(New);
	}
	else if (!find_1 && find_2) { //找到2没找到1，将1带入2
		connection[ind_2].push_back(label_1); //2.pushback[1]
	}
	else if (find_1 && !find_2) {
		connection[ind_1].push_back(label_2);
	}
	else if (find_1 && find_2) { //两个都找到，说明有vector互相融入
		if (ind_1 != ind_2) {
			for (int y = 0; y < connection[ind_2].size(); y++) { //将2带入1
				bool repet = false;
				for (int x = 0; x < connection[ind_1].size(); x++) {
					if (connection[ind_1][x] == connection[ind_2][y]) {
						repet = true;
						break;
					}
				}
				if (!repet) { //如果没有重复
					connection[ind_1].push_back(connection[ind_2][y]);
				}
			}
			connection.erase(connection.begin() + ind_2); //去除2
		}
		else { //ind_1 == ind_2 说明不用管已经规劝了
		}
	}
}

void Label_Update(cv::Mat& label_mask, std::vector<int>& label_cnt) {
	//label更新
	std::vector<std::vector<int>> connection;
	for (int i = 0; i < label_mask.rows - 1; ++i) {
		for (int j = 0; j < label_mask.cols - 1; ++j) {
			int center = label_mask.at<int>(i, j);
			int right = label_mask.at<int>(i, j + 1);
			int down = label_mask.at<int>(i + 1, j);
			if (center != 0 && right != 0 && center != right) {
				Label_Seek(center, right, connection);
			}
			if (center != 0 && down != 0 && center != down) {
				Label_Seek(center, down, connection);
			}
		}
	}

	//构建一个找寻表
	std::vector<int> label_cnt_map;
	for (int i = 0; i < label_cnt.size(); ++i) {
		label_cnt_map.push_back(i);
		label_cnt[i] = 0;
	}
	for (int y = 0; y < connection.size(); y++) {
		for (int x = 0; x < connection[y].size(); x++) {
			label_cnt_map[connection[y][x]] = connection[y][0];
		}
	}

	for (int i = 0; i < label_mask.rows; ++i) {
		for (int j = 0; j < label_mask.cols; ++j) {
			int label = label_mask.at<int>(i, j);
			label_mask.at<int>(i, j) = label_cnt_map[label]; //重新组织

			int real_label = label_mask.at<int>(i, j);
			label_cnt[real_label]++;
		}
	}
}

// 求连通区域
void Connect(const cv::Mat& dst_image, cv::Mat& label_mask, std::vector<int>& label_cnt) {
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
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y, x - 1) == 0) {
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
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y - 1, x) == 0) {
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


void ConnectP4(const cv::Mat& dst_image, cv::Mat& label_mask, std::vector<int>& label_cnt) {
	static std::vector<int> left_neigh; left_neigh.resize(dst_image.rows*dst_image.cols);
	static std::vector<int> up_neigh; up_neigh.resize(dst_image.rows*dst_image.cols);

	for (int y = 0; y < dst_image.rows; y++) {
		
		
		for (int x = 0; x < dst_image.cols; x++) {
			// 左连通
			if (x == 0) {
				left_neigh[y*dst_image.cols+x] = 0;
			}
			else {
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y, x - 1) == 0) {
					left_neigh[y*dst_image.cols+x] = 1;
				}
				else {
					left_neigh[y*dst_image.cols+x] = 0;
				}
			}
			// 上连通
			if (y == 0) {
				up_neigh[y*dst_image.cols+x] = 0;
			}
			else {
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y - 1, x) == 0) {
					up_neigh[y*dst_image.cols+x] = 1;
				}
				else {
					up_neigh[y*dst_image.cols+x] = 0;
				}
			}
		}
	}

	// 维护一个并查集
	int cnt = 1;
	static std::vector<int> connection; connection.clear();
	connection.push_back(0);
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			if (dst_image.at<uchar>(y, x) == 255) {
				label_mask.at<int>(y, x) = 0;
			}
			else {
				bool left = false, up = false;
				if (left_neigh[y*dst_image.cols+x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					left = true;
				}
				if (up_neigh[y*dst_image.cols+x] == 1) {
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
	static std::vector<int> mapping; mapping.clear();
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

