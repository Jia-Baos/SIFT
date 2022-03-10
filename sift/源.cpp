#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "sift.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat image1 = imread("D:\\Code\\picture\\house1.jpg");
	Mat image2 = imread("D:\\Code\\picture\\house2.jpg");
	Mat image01 = image1.clone();
	Mat image02 = image2.clone();

	vector<Keypoint> features1, features2;
	Sift(image01, features1, SIGMA);
	Sift_nointerval(image02, features2, SIGMA);

	//��ͼ���ϻ���������
	DrawKeyPoints(image01, features1);
	DrawKeyPoints(image02, features2);

	//����������������
	DrawSiftFeatures(image01, features1);
	DrawSiftFeatures(image02, features2);

	//���������128ά����
	write_features(features1, "descriptor_1.txt");
	write_features(features2, "descriptor_2.txt");

	imshow("image01", image01);
	imshow("image02", image02);

	/*cv::imwrite("image1_points.jpg", image01);
	cv::imwrite("image2_points.jpg", image02);*/

	//��������ͼ��Ӧ������
	cv::Size size;
	size.width = image1.cols + image2.cols;
	size.height = image1.rows > image2.rows ? image1.rows : image2.rows;
	int width = size.width, height = size.height;

	//��������ͼ��Ϊ���ƶ�Ӧ��������׼��
	Mat img_match(size.height, size.width, CV_8UC3);
	for (int i = 0; i < size.height; i++)
	{
		for (int j = 0; j < size.width; j++)
		{
			//��if��ָֹ�볬��ͼ��ı߽�
			if (i < image1.rows && j < image1.cols)
			{
				img_match.data[(i*width + j) * 3] = image1.data[(i*image1.cols + j) * 3];
				img_match.data[(i*width + j) * 3 + 1] = image1.data[(i*image1.cols + j) * 3 + 1];
				img_match.data[(i*width + j) * 3 + 2] = image1.data[(i*image1.cols + j) * 3 + 2];
			}

			//��if��ָֹ�볬��ͼ��ı߽�
			if (i < image2.rows && j >= image1.cols)
			{
				img_match.data[(i*width + j) * 3] = image2.data[(i*image2.cols + j - image1.cols) * 3];
				img_match.data[(i*width + j) * 3 + 1] = image2.data[(i*image2.cols + j - image1.cols) * 3 + 1];
				img_match.data[(i*width + j) * 3 + 2] = image2.data[(i*image2.cols + j - image1.cols) * 3 + 2];

			}

		}

	}

	//��Ϊͼ���Ƶ����Ҳ࣬���Ժ�����Ҫ�޸�
	cv::Size size_left;
	size_left.height = image1.rows;
	size_left.width = image1.cols;

	//clone�������е�ͼ�񣬻���������
	Mat img_match_keypoints = img_match.clone();
	DrawKeyPoints(img_match_keypoints, features1);
	DrawKeyPointsRight(img_match_keypoints, features2, size_left);
	imshow("img_match_keypoints", img_match_keypoints);


	//�洢��Ӧ��������features�е�����������(1, 9)
	vector<MatchPoint> v_matchpoint;
	Mat img_match_line = img_match.clone();

	//����ͳ��ƥ�������ںʹν�������������˴˾�������������
	int num = 0;

	//������ƥ��
	for (int i = 0; i < features1.size(); i++)
	{
		double odest = 0; //ŷ�Ͼ���
		//��������ֵ���������ʼ��
		int min_index = 0; //��Сֵ������
		double min_value = DBL_MAX; //��Сֵ
		int penuntimatemin_index = 0; //����Сֵ������
		double penuntimatemin_value = DBL_MAX; //����Сֵ
		for (int j = 0; j < features2.size(); j++)
		{
			//����ŷʽ�����
			for (int k = 0; k < FEATURE_ELEMENT_LENGTH; k++)
				odest += (features1[i].descriptor[k] - features2[j].descriptor[k])*(features1[i].descriptor[k] - features2[j].descriptor[k]);

			//odest��ʾ�������������ʸ����ŷ�Ͼ���
			odest = sqrt(odest);

			//�ж��Ƿ�����С
			//�������С������С��ֵ����С�����������С���ж��ǲ��Ǵ�С
			if (odest < min_value)
			{
				penuntimatemin_value = min_value;
				min_value = odest;
				min_index = j;
			}
			else
			{
				if (odest < penuntimatemin_value)
				{
					penuntimatemin_value = odest;
					penuntimatemin_index = j;
				}


			}

		}

		//std::cout << "min_value / penuntimatemin_value: " << min_value / penuntimatemin_value << std::endl;
		
		if (0 < min_index < features2.size() && 0 < penuntimatemin_index < features2.size())
		{
			if (abs(features2[min_index].dx - features2[penuntimatemin_index].dx) < 2)
			{
				std::cout << features2[min_index].dx << std::endl;
				std::cout << features2[min_index].scale << std::endl;
				std::cout << features2[penuntimatemin_index].dx << std::endl;
				std::cout << features2[penuntimatemin_index].scale << std::endl;
				num++;
				std::cout << std::endl;
			}
		}
		//�ж���Сֵ�ʹ�Сֵ�ı�ֵ�Ƿ�С����ֵ
		//С����ֵ����MatchPoint���������Ǽ��ϻ����㣬��λ��������߶Ȳ�ͬ�ĵ�Ҳ������ƥ��㣨���Խ����ƥ���̫���ˣ�
		if (min_value / penuntimatemin_value < 0.5)
		{
			MatchPoint *matchpoint = new MatchPoint;
			matchpoint->index_1 = i;
			matchpoint->index_2 = min_index;
			v_matchpoint.push_back(*matchpoint);
		}

	}

	//ͳ��ƥ�������ںʹν�������������˴˾�������������
	std::cout << num << std::endl;

	//ƥ��������
	std::cout << "match number: " << v_matchpoint.size() << std::endl;

	//������ͼ���Ӧ���������������
	for (int i = 0; i < v_matchpoint.size(); i++)
	{
		Point point1, point2;
		point1.x = features1[v_matchpoint[i].index_1].dx;
		point1.y = features1[v_matchpoint[i].index_1].dy;
		point2.x = features2[v_matchpoint[i].index_2].dx + image01.cols;
		point2.y = features2[v_matchpoint[i].index_2].dy;
		line(img_match_line, point1, point2, cv::Scalar(0, 255, 0));
	}

	imshow("img_match_line", img_match_line);
	/*imwrite("image_match_line.jpg", img_match_line);*/

	waitKey(0);
	return 0;
}
