#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


/******************************* Defs and macros *****************************/

//��������
typedef double pixel_t;

/** default sigma for initial camera smoothing */
#define INIT_SIGMA 0.5

/** default sigma for initial gaussian smoothing */
#define SIGMA 1.6

/** default number of sampled intervals per octave */
#define INTERVALS 3

/** default threshold on keypoint ratio of principle curvatures */
#define RATIO 10

/** maximum steps of keypoint interpolation before failure */
#define MAX_INTERPOLATION_STEPS 5 

/** default threshold on keypoint contrast |D(x)| */
#define DXTHRESHOLD 0.03

/** default number of bins in histogram for orientation assignment */
#define ORI_HIST_BINS 36    

/** determines gaussian sigma for orientation assignment */
#define ORI_SIGMA_TIMES 1.5

/** determines the radius of the region used in orientation assignment */
#define ORI_WINDOW_RADIUS 3.0 * ORI_SIGMA_TIMES 

/** number of passes of orientation histogram smoothing */
#define ORI_SMOOTH_TIMES 2

/** orientation magnitude relative to max that results in new feature */
#define ORI_PEAK_RATIO 0.8

/** length 0f sift feature */
#define FEATURE_ELEMENT_LENGTH 128

/** default number of bins per histogram in descriptor array */
#define DESCR_HIST_BINS 8

/* width of border in which to ignore keypoints */
#define IMG_BORDER 5 

/** default width of descriptor histogram array, 4 * 4 = 16 */
#define DESCR_WINDOW_WIDTH 4

/* determines the size of a single descriptor orientation histogram */
#define DESCR_SCALE_ADJUST 3

/* threshold on magnitude of elements of descriptor vector */
#define DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define INT_DESCR_FCTR 512.0


/******************************** Structures *********************************/

struct Keypoint
{
	int octave;										//�ؼ���������
	int interval;									//�ؼ������ڲ�
	double offset_interval;							//������Ĳ������

	int x;											//x���꣬����octave��interval��ȡ�Ĳ���ͼ��
	int y;											//y���꣬����octave��interval��ȡ�Ĳ���ͼ��


	//scale = sigma0*pow(2.0, o+s/S)
	double scale;									//�ռ�߶����꣬���ڻָ���-1�㣨��0���0�㣩�ĳ߶�
	double dx;										//�������x���꣬�����걻�ָ���-1�㣨��0���0�㣩�ĳ߶�
	double dy;										//�������y���꣬�����걻�ָ���-1�㣨��0���0�㣩�ĳ߶�

	double offset_x;								//�������x���������
	double offset_y;								//�������y���������


	double octave_scale;							//�ؼ�������������ڳ߶�
													//��˹���������ڸ���߶ȣ���ͬ�����ͬ���octave_scaleֵ��ͬ

	double ori;										//�ؼ��㷽��

	int descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];		//128ά����������
	double val;										//��ֵ
};


struct MatchPoint
{
	int index_1;									//ƥ����������ͼ1���±�
	int index_2;									//ƥ����������Сֵ��ͼ2���±�

};


/**************************** function declaration ****************************/

void ConvertToGray(const Mat &src, Mat &dst);

void DownSample(const Mat &src, Mat &dst);

void UpSample(const Mat &src, Mat &dst);

void GaussianTemplateSmooth(const Mat &src, Mat &dst);

void GaussianSmooth2D(const Mat &src, Mat &dst, double sigma);

void GaussianSmooth(const Mat &src, Mat &dst, double sigma);

void CreateInitSmoothGray(const Mat &src, Mat &dst, double);

void GaussianPyramid(const Mat &src, vector<Mat> &gauss_pyr, int octaves, int intervals, double sigma);

void Sub(const Mat &a, const Mat &b, Mat &c);

void DogPyramid(const vector<Mat> &gauss_pyr, vector<Mat> &dog_pyr, int octaves, int intervals);

void DetectionLocalExtrema(const vector<Mat> &dog_pyr, vector<Keypoint> &extrema, int octaves, int intervals);

void DetectionLocalExtrema_nointerval(const vector<Mat> &dog_pyr, vector<Keypoint> &extrema, int octaves, int intervals);

void Sift(const Mat &src, vector<Keypoint> &features, double sigma = SIGMA, int intervals = INTERVALS);

void Sift_nointerval(const Mat &src, vector<Keypoint> &features, double sigma = SIGMA, int intervals = INTERVALS);

void DrawSiftFeatures(Mat &src, vector<Keypoint> &features);

void DrawKeyPoints(Mat &src, vector<Keypoint> &keypoints);

void DrawKeyPointsRight(Mat &src, vector<Keypoint> &keypoints, cv::Size size);

const char *GetFileName(const char* dir, int i);

void write_pyr(const vector<Mat> &pyr, const char* dir);

void display_pyr(const vector<Mat> &pyr, string dir);

void read_features(vector<Keypoint> &features, const char* file);

void write_features(const vector<Keypoint> &features, const char* file);

void testInverse3D();