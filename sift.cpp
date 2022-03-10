//ʹ��printf����
#define _CRT_SECURE_NO_DEPRECATE

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "sift.h"
using namespace std;

/*
  ��ͼ����uchar��RGB��ͨ����ɫͼ��ת��Ϊdouble��һͨ���Ҷ�ͼ��

  @param src ����ͼ��
  @param dst ���ͼ��
*/
void ConvertToGray(const Mat &src, Mat &dst)
{
	Size size = src.size();
	if (dst.empty())
		dst.create(size, CV_64F);

	//����ָ��ָ��src���׵�ַ
	uchar* srcData = src.data;
	//����ָ��ָ��dst���׵�ַ
	pixel_t* dstData = (pixel_t*)dst.data;

	//step[0] ����һ�е����ݴ�С(Byte)
	//step[1] ����һ��Ԫ�ص����ݴ�С(Byte)
	//dstData[0] ����ÿһ��������ռ���ֽ���
	//dstStep ÿ�е����ظ���
	//�������ʿɰ�ͼ����һά���鴦�����ھ���������ø��ӷ���
	int dstStep = dst.step / sizeof(dstData[0]);

	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			double b = *(srcData + src.step * i + src.channels() * j + 0) / 255.0;
			double g = *(srcData + src.step * i + src.channels() * j + 1) / 255.0;
			double r = *(srcData + src.step * i + src.channels() * j + 2) / 255.0;

			*((dstData + dstStep * i + dst.channels() * j)) = b * 0.114 + g * 0.587 + r * 0.299;
		}

	}

}

/*
  ��double��һͨ���Ҷ�ͼ����ж�Ԫ�²���

  @param src ����ͼ��
  @param dst ���ͼ��
*/
void DownSample(const Mat &src, Mat &dst)
{
	if (src.channels() != 1)
		return;

	if (src.cols <= 1 || src.rows <= 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create((int)(src.rows / 2), (int)(src.cols / 2), src.type());

	//����ָ��ָ��src���׵�ַ
	pixel_t* srcData = (pixel_t*)src.data;
	//����ָ��ָ��dst���׵�ַ
	pixel_t* dstData = (pixel_t*)dst.data;

	//srcStep ÿ�е����ظ���
	int srcStep = src.step / sizeof(srcData[0]);
	//dstStep ÿ�е����ظ���
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols; j += 2, n++)
	{
		m = 0;
		for (int i = 0; i < src.rows; i += 2, m++)
		{
			//��ԭͼ����ж�Ԫ�²������������sample
			pixel_t sample = *(srcData + srcStep * i + src.channels() * j);

			//��ֹ��ͼ�񳤿�һ��ʱ������Ϊ����ʱ��m,nԽ��
			if (m < dst.rows && n < dst.cols)
			{
				//����Ԫ�²����Ľ���������dst
				*(dstData + dstStep * m + dst.channels() * n) = sample;

			}

		}

	}

}

/*
  ��double��һͨ���Ҷ�ͼ����ж�Ԫ�ϲ�������ֵ��

  @param src ����ͼ��
  @param dst ���ͼ��
*/
void UpSample(const Mat &src, Mat &dst)
{
	if (src.channels() != 1)
		return;
	dst.create(src.rows * 2, src.cols * 2, src.type());

	//����ָ��ָ��src���׵�ַ
	pixel_t* srcData = (pixel_t*)src.data;
	//����ָ��ָ��dst���׵�ַ
	pixel_t* dstData = (pixel_t*)dst.data;

	//srcStep ÿ�е����ظ���
	int srcStep = src.step / sizeof(srcData[0]);
	//dstStep ÿ�е����ظ���
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols - 1; j++, n += 2)
	{
		m = 0;
		for (int i = 0; i < src.rows - 1; i++, m += 2)
		{
			//�Ƚ�ԭͼ�������ֵcopy��dst��indexΪż����λ����
			double sample = *(srcData + srcStep * i + src.channels() * j);
			*(dstData + dstStep * m + dst.channels() * n) = sample;

			//�в�ֵ
			double rs = *(srcData + srcStep * (i)+src.channels()*j) + (*(srcData + srcStep * (i + 1) + src.channels()*j));
			*(dstData + dstStep * (m + 1) + dst.channels() * n) = rs / 2;

			//�в�ֵ
			double cs = *(srcData + srcStep * i + src.channels()*(j)) + (*(srcData + srcStep * i + src.channels()*(j + 1)));
			*(dstData + dstStep * m + dst.channels() * (n + 1)) = cs / 2;

			//�Խ�λ�ò�ֵ
			double center = (*(srcData + srcStep * (i + 1) + src.channels() * j))
				+ (*(srcData + srcStep * i + src.channels() * j))
				+ (*(srcData + srcStep * (i + 1) + src.channels() * (j + 1)))
				+ (*(srcData + srcStep * i + src.channels() * (j + 1)));

			*(dstData + dstStep * (m + 1) + dst.channels() * (n + 1)) = center / 4;

		}

	}



	if (dst.rows < 3 || dst.cols < 3)
		return;

	//�������
	for (int k = dst.rows - 1; k >= 0; k--)
	{
		*(dstData + dstStep * (k)+dst.channels()*(dst.cols - 2)) = *(dstData + dstStep * (k)+dst.channels()*(dst.cols - 3));
		*(dstData + dstStep * (k)+dst.channels()*(dst.cols - 1)) = *(dstData + dstStep * (k)+dst.channels()*(dst.cols - 3));
	}
	//�������
	for (int k = dst.cols - 1; k >= 0; k--)
	{
		*(dstData + dstStep * (dst.rows - 2) + dst.channels()*(k)) = *(dstData + dstStep * (dst.rows - 3) + dst.channels()*(k));
		*(dstData + dstStep * (dst.rows - 1) + dst.channels()*(k)) = *(dstData + dstStep * (dst.rows - 3) + dst.channels()*(k));
	}

}

/*
  ����7*7��˹����˶�uchar��ͼ�����ƽ��

  @param src ����ͼ��
  @param dst ���ͼ��
*/
void GaussianTemplateSmooth(const Mat &src, Mat &dst)
{
	//��˹ģ��(7*7)��sigma = 0.84089642����һ����õ�
	static const double gaussianTemplate[7][7] =
	{
		{0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
		{0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
		{0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
		{0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771},
		{0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
		{0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
		{0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
	};

	dst.create(src.size(), src.type());

	//����ָ��ָ��src���׵�ַ
	uchar* srcData = src.data;
	//����ָ��ָ��dst���׵�ַ
	uchar* dstData = dst.data;

	for (int j = 0; j < src.cols - 7; j++)
	{
		for (int i = 0; i < src.rows - 7; i++)
		{
			double acc = 0;
			double accb = 0, accg = 0, accr = 0;
			for (int m = 0; m < 7; m++)
			{
				for (int n = 0; n < 7; n++)
				{
					if (src.channels() == 1)
						acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * gaussianTemplate[m][n];
					else
					{
						accb += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 0) * gaussianTemplate[m][n];
						accg += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 1) * gaussianTemplate[m][n];
						accr += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 2) * gaussianTemplate[m][n];
					}
				}
			}
			if (src.channels() == 1)
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3)) = (int)acc;
			else
			{
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 0) = (int)accb;
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 1) = (int)accg;
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 2) = (int)accr;
			}

		}

	}

}

/*
  ����7*7��˹����˶�uchar��һͨ���Ҷ�ͼ�����ƽ��

  @param src ����ͼ��
  @param dst ���ͼ��
  @param sigma ��˹����˵�sqrt(����)
*/
void GaussianSmooth2D(const Mat &src, Mat &dst, double sigma)
{
	if (src.channels() != 1)
		return;

	//ȷ��sigmaΪ���� 
	sigma = sigma > 0 ? sigma : 0;
	//��˹�˾���Ĵ�СΪ(6*sigma+1)*(6*sigma+1)
	//ksizeΪ����
	int ksize = cvRound(sigma * 3) * 2 + 1;

	if (ksize == 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create(src.size(), src.type());

	//�����˹�˾���
	double *kernel = new double[ksize*ksize];

	double scale = -0.5 / (sigma*sigma);
	const double PI = 3.141592653;
	double cons = -scale / PI;

	double sum = 0;

	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			int x = i - (ksize - 1) / 2;
			int y = j - (ksize - 1) / 2;
			kernel[i*ksize + j] = cons * exp(scale * (x*x + y * y));

			sum += kernel[i*ksize + j];
		}
	}
	//��˹�˹�һ��
	for (int i = ksize * ksize - 1; i >= 0; i--)
	{
		*(kernel + i) /= sum;
	}

	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	//ͼ��������
	for (int j = 0; j < src.cols - ksize; j++)
	{
		for (int i = 0; i < src.rows - ksize; i++)
		{
			double acc = 0;

			for (int m = 0; m < ksize; m++)
			{
				for (int n = 0; n < ksize; n++)
				{
					acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * kernel[m*ksize + n];
				}
			}

			*(dstData + dst.step * (i + (ksize - 1) / 2) + (j + (ksize - 1) / 2)) = (int)acc;
		}

	}
	delete[]kernel;
}

/*
  ����opencv�ٷ�������ͼ�����ƽ��

  @param src ����ͼ��
  @param dst ���ͼ��
  @param sigma ��˹����˵�sqrt(����)
*/
void GaussianSmooth(const Mat &src, Mat &dst, double sigma)
{
	GaussianBlur(src, dst, Size(0, 0), sigma, sigma);
}

/*
  ������0���0��ͼ�����������߶�Ϊsigma

  @param src ����ͼ��
  @param dst ���ͼ��
  @param sigma -1�㣨��0���0�㣩�ĳ߶�
*/
void CreateInitSmoothGray(const Mat &src, Mat &dst, double sigma = SIGMA)
{
	Mat gray, up;

	ConvertToGray(src, gray);
	UpSample(gray, up);

	//sigma��-1��ĳ߶ȣ�sigma_init�Ǹ�˹����˵�sprt(����)
	double  sigma_init = sqrt(sigma * sigma - (INIT_SIGMA * 2) * (INIT_SIGMA * 2));

	GaussianSmooth(up, dst, sigma_init);
}

/*
  ������˹�߶ȿռ�

  @param src ����ͼ��
  @param gauss_pyr ��˹������
  @param octaves ��˹������������
  @param intervals ��Ч��ֵ����Ĳ���
  @param sigma -1�㣨��0���0�㣩�ĳ߶�
*/
void GaussianPyramid(const Mat &src, vector<Mat> &gauss_pyr, int octaves, int intervals = INTERVALS, double sigma = SIGMA)
{
	//����洢���ڲ���˹ƽ������Ҫ�����ӵ����飬֮��ÿһ�鶼������һ����ȥ���ɣ���ϸ��һ��ԭ�򣡣���
	//sqrt(sig_total * sig_total - sig_prev * sig_prev)
	double *sigmas = new double[intervals + 3];
	double k = pow(2.0, 1.0 / intervals);

	sigmas[0] = sigma;

	double sig_prev, sig_total;
	for (int i = 1; i < intervals + 3; i++)
	{
		sig_prev = pow(k, i - 1) * sigma;
		sig_total = sig_prev * k;
		sigmas[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	for (int o = 0; o < octaves; o++)
	{
		//ÿ�������
		for (int i = 0; i < intervals + 3; i++)
		{
			Mat mat;
			if (o == 0 && i == 0)
			{
				src.copyTo(mat);
			}
			else if (i == 0)
			{
				//			DownSample(gauss_pyr[o*(intervals+3)-2], mat);  //error
								//ǰһ���˹ͼ��ĵ��������㣨�������Ĳ㣩
								//��ͼ���±�Ϊ��
								//0 1 2 3  4  5  //o=0
								//6 7 8 9 10 11  //o=1
								//...
								//��һ���һ��ͼ(�±�Ϊ6)��ͼ����0���±�Ϊ3��ͼ�񽵲�������
				DownSample(gauss_pyr[(o - 1)*(intervals + 3) + intervals], mat);
			}
			else
			{
				//ÿ������һ������һ���˹ģ���õ������ø�˹����˵Ĵ�СΪsqrt(sig_late * sig_late - sig_prev * sig_prev)��
				GaussianSmooth(gauss_pyr[o * (intervals + 3) + i - 1], mat, sigmas[i]);
			}
			gauss_pyr.push_back(mat);

		}

	}
	delete[] sigmas;

}

/*
  �ɸ�˹������ͬһ�����������ͼ�����ɲ��ͼ��

  @param a ����ͼ��1
  @param b ����ͼ��2
  @param c ���ͼ�񣨲��ͼ��
*/
void Sub(const Mat &a, const Mat &b, Mat &c)
{
	if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
		return;
	if (!c.empty())
		return;
	c.create(a.size(), a.type());

	pixel_t* ap = (pixel_t*)a.data;
	pixel_t* ap_end = (pixel_t*)a.dataend;
	pixel_t* bp = (pixel_t*)b.data;
	pixel_t* cp = (pixel_t*)c.data;

	int step = a.step / sizeof(pixel_t);

	//while��ʹ��ʹ������Ӽ�ֵ࣬��ѧϰ
	while (ap != ap_end)
	{
		*cp++ = *ap++ - *bp++;
	}

}

/*
  ���ɸ�˹��ֽ�����

  @param gauss_pyr ��˹������
  @param dog_pyr ��˹��ֽ�����
  @param octaves ��˹��ֽ�����������
  @param intervals ��Ч��ֵ����Ĳ���
*/
void DogPyramid(const vector<Mat> &gauss_pyr, vector<Mat> &dog_pyr, int octaves, int intervals = INTERVALS)
{
	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < intervals + 3; i++)
		{
			Mat mat;
			//���ɵĲ��ͼ��ĳ߶���gauss_pyr[o*(intervals + 3) + i - 1]�ĳ߶���һ���ģ�������Ҫ������
			Sub(gauss_pyr[o*(intervals + 3) + i], gauss_pyr[o*(intervals + 3) + i - 1], mat);
			dog_pyr.push_back(mat);
		}

	}

}


/*
  �ж�ĳ����������Χ3*3*3�������Ƿ�Ϊ��ֵ

  @param x ���ص�x���꣨ͼ������ϵ�£�
  @param y ���ص�y���꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @return ����Ǽ�ֵ�򷵻�true�������򷵻�false
*/
bool isExtremum(int x, int y, const vector<Mat> &dog_pyr, int index)
{
	pixel_t * data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	if (val > 0)
	{
		//������
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			//������
			for (int j = -1; j <= 1; j++)
			{
				//������
				for (int k = -1; k <= 1; k++)
				{
					//������ֵ
					if (val < *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		//������
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			//int stp = dog_pyr[index + i].step / sizeof(pixel_t);
			//��������һ�д�������һ�������ǻ�ȡԪ�ص��ֽ�����һ����ֱ�ӷ���Ԫ�أ�һ���Ƿ���double����

			//������
			for (int j = -1; j <= 1; j++)
			{
				//������
				for (int k = -1; k <= 1; k++)
				{
					//�����С��ֵ
					if (val > *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}

	return true;
}

bool isExtremum_nointerval(int x, int y, const vector<Mat> &dog_pyr, int index)
{
	pixel_t * data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	if (val > 0)
	{

		int stp = dog_pyr[index].step / sizeof(data[0]);
		//������
		for (int j = -1; j <= 1; j++)
		{
			//������
			for (int k = -1; k <= 1; k++)
			{
				//������ֵ
				if (val < *((pixel_t*)dog_pyr[index].data + stp * (y + j) + (x + k)))
				{
					return false;
				}

			}

		}

	}

	else
	{
		int stp = dog_pyr[index].step / sizeof(data[0]);
		//int stp = dog_pyr[index + i].step / sizeof(pixel_t);
		//��������һ�д�������һ�������ǻ�ȡԪ�ص��ֽ�����һ����ֱ�ӷ���Ԫ�أ�һ���Ƿ���double����

		//������
		for (int j = -1; j <= 1; j++)
		{
			//������
			for (int k = -1; k <= 1; k++)
			{
				//�����С��ֵ
				if (val > *((pixel_t*)dog_pyr[index].data + stp * (y + j) + (x + k)))
				{
					return false;
				}

			}

		}

	}

	return true;
}

//����ͼ���x��yλ�õ����أ�x��ʾ�����꣬y��ʾ�����꣩
#define DAt(x, y) (*(data+(y)*step+(x)))

/*
  ������Ե��Ӧ
  Tr_h * Tr_h / Det_h��ֵ����������ֵ���ʱ��С��ֵԽ��˵����������ֵ�ı�ֵԽ��
  ����ĳһ��������ݶ�ֵԽ�󣬶�����һ��������ݶ�ֵԽС������Եǡǡ�������������
  ����Ϊ���޳���Ե��Ӧ�㣬��Ҫ�øñ�ֵС��һ������ֵ��

  @param x ���صĺ�����
  @param y ���ص�������
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param r ������Ե��Ӧ�������ֵ��Lowe����Ϊ10

  @return ���С����ֵ�򷵻�true�����򷵻�false
*/
bool passEdgeResponse(int x, int y, const vector<Mat> &dog_pyr, int index, double r = RATIO)
{
	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	double Dxx, Dyy, Dxy;
	double Tr_h, Det_h;

	//hessian����
	//	   _ 	    _
	//    | Dxx  Dxy |
	// H =|			 |
	//	  |_Dxy  Dyy_|	
	//	  
	Dxx = DAt(x + 1, y) + DAt(x - 1, y) - 2 * val;
	Dyy = DAt(x, y + 1) + DAt(x, y - 1) - 2 * val;
	Dxy = (DAt(x + 1, y + 1) + DAt(x - 1, y - 1) - DAt(x - 1, y + 1) - DAt(x + 1, y - 1)) / 4.0;

	Tr_h = Dxx + Dyy;
	Det_h = Dxx * Dyy - Dxy * Dxy;

	if (Det_h <= 0)
		return false;

	//С����ֵ����Ϊ���ǹؼ���
	if (Tr_h * Tr_h / Det_h < (r + 1) * (r + 1) / r)
		return true;

	return false;
}

//���ʽ���������Ϊindex���x��yλ�õ�����
double PyrAt(const vector<Mat> &pyr, int index, int x, int y)
{
	pixel_t *data = (pixel_t*)pyr[index].data;
	int step = pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return val;
}

#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))

/*
  ����һ�׵������飨1*3��

  @param x ���صĺ����꣨ͼ������ϵ�£�
  @param y ���ص������꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param dx �洢һ�׵���������
*/
void DerivativeOf3D(int x, int y, const vector<Mat> &dog_pyr, int index, double *dx)
{
	double Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	double Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	double Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;
}

//����Hessian�����i��jλ�õ�Ԫ��
//Hessian����������һά����洢
#define Hat(i, j) (*(H+(i)*3 + (j)))

/*
  ����Hessian����3*3��

  @param x ���صĺ����꣨ͼ������ϵ�£�
  @param y ���ص������꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param H �洢Hessian���������
*/
void Hessian3D(int x, int y, const vector<Mat> &dog_pyr, int index, double *H)
{
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;

	val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1)
		- At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;

	Dxs = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y)
		- At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;

	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1)
		- At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;

	Hat(0, 0) = Dxx;
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;

	Hat(1, 0) = Hat(0, 1) = Dxy;
	Hat(2, 0) = Hat(0, 2) = Dxs;
	Hat(2, 1) = Hat(1, 2) = Dys;
}

//����Hessian��������i��jλ�õ�Ԫ��
//Hessian��������������һά����洢
#define HIat(i, j) (*(H_inve+(i)*3 + (j)))

/*
  ����Hessian������棨3*3��

  @param x ���صĺ����꣨ͼ������ϵ�£�
  @param y ���ص������꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param H_inve �洢Hessian������������

  @return ������������򷵻�true�����򷵻�false
*/
bool Inverse3D(const double *H, double *H_inve)
{
	//A=|H|
	//		 / A00 A01 A02 \				   
	//��H =  | A10 A11 A12 |   
	//		 \ A20 A21 A22 /	
	//�� ����ʽ|H|=A00*A11*A22+A01*A12*A20+A02*A10*A21
	//	    -A00*A12*A21-A01*A10*A22-A02*A11*A20
	//

	double A = Hat(0, 0)*Hat(1, 1)*Hat(2, 2)
		+ Hat(0, 1)*Hat(1, 2)*Hat(2, 0)
		+ Hat(0, 2)*Hat(1, 0)*Hat(2, 1)
		- Hat(0, 0)*Hat(1, 2)*Hat(2, 1)
		- Hat(0, 1)*Hat(1, 0)*Hat(2, 2)
		- Hat(0, 2)*Hat(1, 1)*Hat(2, 0);

	//û�������
	if (fabs(A) < 1e-10)
		return false;

	//������������㹫ʽ��������󷨣���
	//		 / a b c \				    / ei-hf -(bi-ch) bf-ce\
	//��A =  | d e f |   ��A(-1) =1/|H|*| fg-id -(cg-ia) cd-af |
	//		 \ g h i /				    \ dh-ge -(ah-gb) ae-bd/

	HIat(0, 0) = Hat(1, 1) * Hat(2, 2) - Hat(2, 1)*Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1) * Hat(2, 2) - Hat(2, 1) * Hat(0, 2));
	HIat(0, 2) = Hat(0, 1) * Hat(1, 2) - Hat(0, 2)*Hat(1, 1);

	HIat(1, 0) = Hat(1, 2) * Hat(2, 0) - Hat(2, 2)*Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2) * Hat(2, 0) - Hat(0, 0) * Hat(2, 2));
	HIat(1, 2) = Hat(0, 2) * Hat(1, 0) - Hat(0, 0)*Hat(1, 2);

	HIat(2, 0) = Hat(1, 0) * Hat(2, 1) - Hat(1, 1)*Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0) * Hat(2, 1) - Hat(0, 1) * Hat(2, 0));
	HIat(2, 2) = Hat(0, 0) * Hat(1, 1) - Hat(0, 1)*Hat(1, 0);

	for (int i = 0; i < 9; i++)
	{
		//i = 0ʱ��ָ��ָ������ĳ�ʼλ��
		*(H_inve + i) /= A;
	}
	return true;
}

/*
  ���������ؼ���ֵ�������ƫ���������飨1*3��

  @param x ���صĺ����꣨ͼ������ϵ�£�
  @param y ���ص������꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param offset_x �洢�����ؼ���ֵ�������ƫ����������
*/
void GetOffsetX(int x, int y, const vector<Mat> &dog_pyr, int index, double *offset_x)
{
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double H[9], H_inve[9] = { 0 };
	Hessian3D(x, y, dog_pyr, index, H);
	Inverse3D(H, H_inve);
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	for (int i = 0; i < 3; i++)
	{
		offset_x[i] = 0.0;
		//�Ⱦ�������
		for (int j = 0; j < 3; j++)
		{
			offset_x[i] += H_inve[i * 3 + j] * dx[j];
		}
		//�����෴��
		offset_x[i] = -offset_x[i];
	}
}

/*
  ����Dx���������жԱȶ��ж�

  @param x ���صĺ����꣨ͼ������ϵ�£�
  @param y ���ص������꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param offset_x �洢�����ؼ���ֵ�������ƫ����������

  @return Dx
*/
double GetFabsDx(int x, int y, const vector<Mat> &dog_pyr, int index, const double *offset_x)
{
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return fabs(val + 0.5 * term);
}

/*
  ���������ؼ���ֵ

  @param x ���صĺ����꣨ͼ������ϵ�£�
  @param y ���ص������꣨ͼ������ϵ�£�
  @param dog_pyr ��˹��ֽ�����
  @param index ����������ͼ�������
  @param ocatave ��ֵ�����ڵ���
  @param interval ��ֵ��������Ĳ�
  @param dxthreshold �ж�Dx�õ���ֵ��|D(x)| < 0.03 Lowe 2004

  @return Keypoint �ؼ����������Ϣ
*/
Keypoint *InterploationExtremum(int x, int y, const vector<Mat> &dog_pyr, int index, int octave, int interval, double dxthreshold = DXTHRESHOLD)
{

	double offset_x[3] = { 0 };
	//��dog_pyr[index]����Ϊconst����ֹ���ܸ�
	const Mat &mat = dog_pyr[index];

	//��ǰͼ�����ڵ�����������octave��interval�������
	int idx = index;

	//��ǰͼ�㴦�ڵ�ǰ�����һ��
	int intvl = interval;
	//��ֵ��������
	int i = 0;
	while (i < MAX_INTERPOLATION_STEPS)
	{
		GetOffsetX(x, y, dog_pyr, idx, offset_x);
		//Accurate keypoint localization.  Lowe
		//���offset_x ����һά�ȴ���0.5��it means that the extremum lies closer to a different sample point.
		if (fabs(offset_x[0]) < 0.5 && fabs(offset_x[1]) < 0.5 && fabs(offset_x[2]) < 0.5)
			//���̩�ղ�ֵ��κ����������ƫ������С��0.5��˵���Ѿ��ҵ������㣬���˳�����
			break;

		//������õ���ƫ�������¶����ֵ���ĵ�����λ��
		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		interval += cvRound(offset_x[2]);

		//��������idx������ʹ��idx������DoG������
		idx = index - intvl + interval;

		//�˴���֤����ʱ x+1,y+1��x-1, y-1��Ч
		if (interval < 1 || interval > INTERVALS ||
			x >= mat.cols - 1 || x < 2 ||
			y >= mat.rows - 1 || y < 2)
		{
			return NULL;
		}

		i++;
	}

	//�ܸ�ʧ�ܣ�ȷ���Ƿ���ڵ�������
	if (i >= MAX_INTERPOLATION_STEPS)
		return NULL;

	//rejecting unstable extrema���޳����ȶ��ļ�ֵ��
	//|D(x^)| < 0.03ȡ����ֵ����ʱ�������ǳ���
	if (GetFabsDx(x, y, dog_pyr, idx, offset_x) < dxthreshold / INTERVALS)
	{
		return NULL;
	}

	//����ǰ�������������
	Keypoint *keypoint = new Keypoint;


	keypoint->x = x;
	keypoint->y = y;

	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];

	keypoint->interval = interval;
	keypoint->offset_interval = offset_x[2];

	keypoint->octave = octave;

	keypoint->dx = (x + offset_x[0])*pow(2.0, octave);
	keypoint->dy = (y + offset_x[1])*pow(2.0, octave);

	return keypoint;
}

/*
  ���м�ֵ��ļ�⣬�޳��ͶԱȶȵĵ㣬�ж��Ƿ��Ǽ�ֵ�㣬�����������ؼ���ֵ���޳���Ե��Ӧ��

  @param dog_pyr ��˹��ֽ�����
  @param extrma �洢�ؼ����vector
  @param ocataves ��˹��ֽ���������
  @param intervals ��Ч��ֵ����Ĳ���
*/
void DetectionLocalExtrema(const vector<Mat> &dog_pyr, vector<Keypoint> &extrema, int octaves, int intervals = INTERVALS)
{
	long int dd = 0, cc1 = 0, cc2 = 0, cc3 = 0, cc0 = 0, cc00 = 0;

	double thresh = 0.5 * DXTHRESHOLD / intervals;
	for (int o = 0; o < octaves; o++)
	{
		//��һ������һ�㼫ֵ���ԣ������ٴ�˼���⼸��ĳ߶ȣ�����
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			//����ǰͼ���������octave��interval��ʾ����
			int index = o * (intervals + 2) + i;

			pixel_t *data = (pixel_t *)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);

			for (int y = IMG_BORDER; y < dog_pyr[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dog_pyr[index].cols - IMG_BORDER; x++)
				{
					//ͳ���ж��˶��ٸ����ص�?
					cc00++;

					pixel_t val = *(data + y * step + x);
					//�޳�С����ֵ�ĵ�
					if (fabs(val) > thresh)
					{
						//ͳ���ж��˶��ٸ����ص���ڸ�������ֵ?
						cc0++;
						if (isExtremum(x, y, dog_pyr, index))
						{
							//ͳ���ж��˶��ٸ����ص��Ǿֲ�����ֵ?
							cc1++;
							Keypoint *extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);

							if (extrmum)
							{
								//ͳ���ж��˶��ٸ����ص�����������ؼ���ֵ?
								cc2++;

								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y*step + extrmum->x);

									//ͳ���ж��˶��ٸ����ص������������Ե����Ӧ?
									cc3++;
									extrema.push_back(*extrmum);
								}

								delete extrmum;

							}

						}

					}

				}

			}

		}

	}
	std::cout << "cc00: " << cc00 << ", cc0: " << cc0 << ", cc1: " << cc1 << ", cc2: " <<
		cc2 << ", cc3: " << cc3 << std::endl;
	std::cout << "0.5 * DXTHRESHOLD / intervals: " << thresh << std::endl;
}

void DetectionLocalExtrema_nointerval(const vector<Mat> &dog_pyr, vector<Keypoint> &extrema, int octaves, int intervals = INTERVALS)
{
	long int dd = 0, cc1 = 0, cc2 = 0, cc3 = 0, cc0 = 0, cc00 = 0;

	double thresh = 0.5 * DXTHRESHOLD / intervals;
	for (int o = 0; o < octaves; o++)
	{
		//��һ������һ�㼫ֵ���ԣ������ٴ�˼���⼸��ĳ߶ȣ�����
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			//����ǰͼ���������octave��interval��ʾ����
			int index = o * (intervals + 2) + i;

			pixel_t *data = (pixel_t *)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);

			for (int y = IMG_BORDER; y < dog_pyr[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dog_pyr[index].cols - IMG_BORDER; x++)
				{
					//ͳ���ж��˶��ٸ����ص�?
					cc00++;

					pixel_t val = *(data + y * step + x);
					//�޳�С����ֵ�ĵ�
					if (fabs(val) > thresh)
					{
						//ͳ���ж��˶��ٸ����ص���ڸ�������ֵ?
						cc0++;
						if (isExtremum_nointerval(x, y, dog_pyr, index))
						{
							//ͳ���ж��˶��ٸ����ص��Ǿֲ�����ֵ?
							cc1++;
							Keypoint *extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);

							if (extrmum)
							{
								//ͳ���ж��˶��ٸ����ص�����������ؼ���ֵ?
								cc2++;

								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y*step + extrmum->x);

									//ͳ���ж��˶��ٸ����ص������������Ե����Ӧ?
									cc3++;
									extrema.push_back(*extrmum);
								}

								delete extrmum;

							}

						}

					}

				}

			}

		}

	}
	std::cout << "cc00: " << cc00 << ", cc0: " << cc0 << ", cc1: " << cc1 << ", cc2: " <<
		cc2 << ", cc3: " << cc3 << std::endl;
	std::cout << "0.5 * DXTHRESHOLD / intervals: " << thresh << std::endl;
}

/*
  ���㼫ֵ�����ڲ�ĳ߶�

  @param features �洢�ؼ����vector
  @param sigam -1�㣨��0���0�㣩�ĳ߶�
  @param intervals ��Ч��ֵ����Ĳ���
*/
void CalculateScale(vector<Keypoint> &features, double sigma = SIGMA, int intervals = INTERVALS)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		//����ڵ�0���0��ĳ߶ȣ����ڻָ��ؼ�������ͳ߶�
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		//������������0��ĳ߶ȣ�����ȷ��ͳ�ƹؼ����ݶȷ���ֱ��ͼ������İ뾶
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
	}

}

/*
  �������ͼ������������ţ����ŵ�����ͼ��ĳ߶�

  @param features �洢�������vector

*/
void HalfFeatures(vector<Keypoint> &features)
{
	for (int i = 0; i < features.size(); i++)
	{
		//dx��dy������������ڵ�0���0�������ϵ����Ҫ�ָ�������ͼ��ĳ߶�
		features[i].dx /= 2;
		features[i].dy /= 2;

		//scale������������ڵ�0���0��ĳ߶ȣ���Ҫ�ָ�������ͼ��ĳ߶�
		features[i].scale /= 2;
	}
}

/*
  ����gaussͼ����x��y���ص�ķ�ֵ���ݶȷ���

  @param gauss ��˹�������е�ͼ��
  @param x ���ص�x���꣨ͼ������ϵ�£�
  @param y ���ص�y���꣨ͼ������ϵ�£�
  @param mag ��ֵ
  @param ori �ݶȷ���

  @return û�д�������Ե�򷵻�true�����򷵻�false
*/
bool CalcGradMagOri(const Mat &gauss, int x, int y, double &mag, double &ori)
{
	if (x > 0 && x < gauss.cols - 1 && y > 0 && y < gauss.rows - 1)
	{
		pixel_t *data = (pixel_t*)gauss.data;
		int step = gauss.step / sizeof(*data);

		double dx = *(data + step * y + (x + 1)) - (*(data + step * y + (x - 1)));
		double dy = *(data + step * (y + 1) + x) - (*(data + step * (y - 1) + x));

		mag = sqrt(dx*dx + dy * dy);

		//atan2����[-Pi, -Pi]�Ļ���ֵ
		//
		//				0.5*PI
		//				|
		//		PI		|
		//     ----------------- 0
		//		-PI		|
		//				|
		//				-0.5*PI
		//
		ori = atan2(dy, dx);
		return true;
	}
	else
		return false;
}

/*
  ���㼫ֵ����ݶȷ���ֱ��ͼ

  @param gauss ��˹�������е�ͼ��
  @param x ���ص�x���꣨ͼ������ϵ�£�
  @param y ���ص�y���꣨ͼ������ϵ�£�
  @param bins �ݶȷ���ֱ��ͼ��bins
  @param radius ͳ���ݶȷ���ֱ��ͼ����Ҫ������뾶

  @return �����ݶȷ���ֱ��ͼ
*/
double *CalculateOrientationHistogram(const Mat &gauss, int x, int y, int bins, int radius, double sigma)
{
	//һ��Ϊ36bin
	double *hist = new double[bins];

	//�ݶȷ���ֱ��ͼ����ֵ
	for (int i = 0; i < bins; i++)
		*(hist + i) = 0.0;

	double mag, ori;
	double weight;

	int bin;
	const double PI2 = 2.0*CV_PI;

	double econs = -1.0 / (2.0*sigma*sigma);

	//radius: cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale) = cvRound(3*1.5*extrema[i].octave_scale)
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))
			{
				//sigam: ORI_SIGMA_TIMES*extrema[i].octave_scale = 1.5*extrema[i].octave_scale
				weight = exp((i*i + j * j)*econs);

				//ʹ��Pi-ori��oriת����[0,2*PI]֮��
				//
				//
				//	ת��ǰ��									ת����
				//
				//				0.5*PI									0.5*PI	
				//				|										|
				//		PI		|								0		|
				//     ----------------- 0						----------------- PI
				//		-PI		|								2*PI	|
				//				|										|
				//				-0.5*PI									1.5*PI
				//
				//
				bin = cvRound(bins * (CV_PI - ori) / PI2);
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;
			}
		}
	}

	return hist;
}

/*
  ���ݶȷ���ֱ��ͼ����ƽ��
  ��˹ƽ����ģ��Ϊ{0.25, 0.5, 0.25}
  �������Ϊ��ԭ����hist��ƽ�����ֵ�����µ�hist������ʵ�ַ������������

  @param hist �ݶȷ���ֱ��ͼ
  @param n �ݶȷ���ֱ��ͼ��bins
*/
void GaussSmoothOriHist(double *hist, int n)
{
	double prev = hist[n - 1], temp, h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * (i + 1 >= n ? h0 : hist[i + 1]);
		prev = temp;
	}

}

/*
  ���ݶȷ���ֱ��ͼ��������

  @param hist �ݶȷ���ֱ��ͼ
  @param n �ݶȷ���ֱ��ͼ��bins

  @return ��������bin�е����ֵ
*/
double DominantDirection(double *hist, int n)
{
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)
			maxd = hist[i];
	}
	return maxd;
}

/*
  ����������
  һ����������ָ�����ʱ��ҲҪ����һ���µ������㣬���ֱ��copyһ�ݣ����޸ķ��򣬲������

  @param src ����������
  @param dst ���������
*/
void CopyKeypoint(const Keypoint &src, Keypoint &dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.val = src.val;
	dst.x = src.x;
	dst.y = src.y;
}

//�����߲�ֵ
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 

/*
  ��ƽ������ݶȷ���ֱ��ͼ���������߲�ֵ����ȡ����׼ȷ�ķ���

  @param keypoint ����������
  @param features ����������vector
  @param hist �ݶȷ���ֱ��ͼ
  @param mag_thr �ݶȷ���ֱ��ͼ�������ĳ��bin��ֵ������binֵ��80%��������γɸ�����
*/
void CalcOriFeatures(const Keypoint &keypoint, vector<Keypoint> &features, const double *hist, int n, double mag_thr)
{
	double bin, PI2 = CV_PI * 2.0;

	int l, r;
	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		//hist[i]�Ǽ�ֵ�����ڴ���������80%����ֵ��bin��Ҫ�϶�Ϊ������
		//mag_thr: highest_peak*ORI_PEAK_RATIO = 0.8*highest_peak
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			//���������
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);

			//new_key��keypointֻ�Ƿ���һ��
			Keypoint new_key;
			CopyKeypoint(keypoint, new_key);

			//�����򡢸����򶼿��ǽ�ȥ���γ��µ������㣬��������������Ҫ���ڹؼ��������
			new_key.ori = ((PI2 * bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}


/*
  ����������з�����䣬������������ݶȷ���ֱ��ͼ�������ݶȷ���ֱ��ͼ��ƽ����ȷ��������ȷ��������ľ�ȷ�ǶȺ͸�����ľ�ȷ�Ƕ�

  @param extreme ����������vector
  @param features ����������vector
  @param gauss_pyr ��˹������
*/
void OrientationAssignment(vector<Keypoint> &extrema, vector<Keypoint> &features, const vector<Mat> &gauss_pyr)
{
	int n = extrema.size();
	double *hist;

	for (int i = 0; i < n; i++)
	{

		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave*(INTERVALS + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, ORI_HIST_BINS, cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale),
			ORI_SIGMA_TIMES*extrema[i].octave_scale);

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, ORI_HIST_BINS);

		double highest_peak = DominantDirection(hist, ORI_HIST_BINS);

		CalcOriFeatures(extrema[i], features, hist, ORI_HIST_BINS, highest_peak*ORI_PEAK_RATIO);

		delete[] hist;
	}

}

/*
  �����Բ�ֵ

  @param hist �������������
  @param xbin ���ص�x���꣨ͼ������ϵ�£�
  @param ybin ���ص�y���꣨ͼ������ϵ�£�
  @param obin ���ص�index
  @param mag �ݶȷ���
  @param bins DESCR_HIST_BINS = 8
  @param d DESCR_WINDOW_WIDTH = 4
*/
void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	//������������(0, 0, 0)
	int r0, c0, o0;

	//������������(0, 0, 0)��Խ�λ�õĵ�(1, 1, 1)
	int rb, cb, ob;

	//�������ص�������������ϵ�е�λ�ã�ȡֵ��Χ[0, 1]
	double d_r, d_c, d_o;

	//��ʱ����������ʾ���ص��ĳ�������Ȩ�أ�ע��������һ��һ�������
	//Ȩ�������ɷ��ȣ�weight(r) = 1 - r 
	double v_r, v_c, v_o;

	//��ʱ������������������İ˸�����
	int r, c, o;

	double** row, *h;

	//����ȡ����r0��c0��o0Ϊ��ά������������֣���ʾ�����ĸ�������
	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);

	//d_r��d_c��d_oΪ��ά�����С�����֣�����������C�������
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	/*
		����ֵ��
		xbin,ybin,obin:���ӵ������Ӵ��ڵ�λ�úͷ���
		�������ӵ㶼������4*4�Ĵ�����
		r0,c0ȡ������xbin��ybin��������
		r0,c0ֻ��ȡ��0,1,2
		xbin,ybin��(-1, 2)

		r0ȡ������xbin��������ʱ��
		r0+0 <= xbin <= r0+1
		mag������[r0,r1]������ֵ

		obinͬ��
	*/

	for (r = 0; r <= 1; r++)
	{
		//���������������İ˸����㣬����㣨-0.5��-0.5��6.3��
		//���䲻�ؼ���ԣ�-1��-1�����������Ȩ�أ�ֻ�����ԣ�0��0���������Ȩ��
		//�ֻ�����㣨 3.5�� 3.5��6.3��
		//���䲻�ؼ���ԣ�4��4�����������Ȩ�أ�ֻ�����ԣ�3��3���������Ȩ��
		rb = r0 + r;
		//�ж�row�������Ƿ񳬳�ͳ������rb = {0, 1, 2, 3}
		if (rb >= 0 && rb < d)
		{
			//���row��Ȩ�أ�r == 0����������Ͻǵ��Ȩ�أ�r != 0����������½ǵĵ��Ȩ�أ�Ȩ�������ɷ���
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				//�ж�col�������Ƿ񳬳�ͳ������cb = {0, 1, 2, 3}
				if (cb >= 0 && cb < d)
				{
					//���row��Ȩ��*���col��Ȩ��
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						//��Ϊ�ݶȷ�����ѭ���ģ�����ֱ����ȡ��ķ���
						//����㣨-0.5��-0.5��7.2������� 7 �� 0 ori�㶼�����Ȩ��
						//���Բ��ö�������Ƿ��޵��ж�
						ob = (o0 + o) % bins;
						//���row��Ȩ��*���col��Ȩ��*���ori��Ȩ��
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}

				}

			}

		}

	}

}

/*
  �����������������

  @param gauss gaussͼ��
  @param x ���ص�x���꣨ͼ������ϵ�£�
  @param y ���ص�y���꣨ͼ������ϵ�£�
  @param octave_scale gaussͼ����������ڵ�0��ĳ߶�
  @param ori �������������������Ҫ��ת�ĽǶ�
  @param bins ����������ݶȷ���ֱ��ͼ��bins
  @param width ���򻮷�Ϊ4*4�ĸ�����

  ��ʵ��Ӧ���У�����������������ΪԲ�ģ���(DESCR_SCALE_ADJUST * octave_scale*sqrt(2.0) * (width + 1)) / 2.0Ϊ�뾶��
  �����Բ���������ص��ݶȷ��Ǻ͸�˹��Ȩ����ݶȷ�ֵ��Ȼ���ٵõ���Щ��ֵ�ͷ�������Ӧ����������ת�Ժ��µ�����λ�á�
*/
double ***CalculateDescrHist(const Mat &gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double ***hist = new double**[width];

	//����ռ䲢��ʼ����4*4*8����
	for (int i = 0; i < width; i++)
	{
		hist[i] = new double*[width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;


	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	//��˹Ȩֵ��sigma���������Ӵ��ڿ�ȣ�4����һ��
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma*sigma);

	double PI2 = CV_PI * 2;

	//ÿ��������(�������ڵ�С����)�Ŀ�
	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	//����뾶��+0.5ȡ�������룬ʵ���ϲ���Ҫ����Ϊ����0.5�������������������Ѿ��㹻���Ѿ���������ת������Ӱ�죩
	//int radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0 + 0.5;
	int radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0;

	double grad_ori, grad_mag;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			//˳ʱ����ת��������Ҫ����ĵ��Ѿ�������4*4�����У���������ϵ��ԭ�㻹����������
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;


			//xbin,ybinΪ����4*4���������ص��±�ֵ������ϵ��ԭ���ƶ���4*4���ڵ����Ͻǵ����ӵ���
			double xbin = rot_x + width / 2 - 0.5;
			double ybin = rot_y + width / 2 - 0.5;

			//��ͳ����������Ϊ���ĵ�5*5������������ص㣬�����ӱ���Ĵ�������������Ϊ���ĵ�4*4������
			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				//������������Ϊ���ģ�radiusΪ�뾶Բ���������������صķ�ֵ���ݶȷ���
				//����תǰͳ�Ʒ�ֵ���ݶȷ���
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori))
				{
					//ת���˽Ƕ�����ϵ�����ҽ�����ת����תori������ĽǶȣ�����ϸ�ο�1051 ~ 1064�еĴ���
					grad_ori = (CV_PI - grad_ori) - ori;

					//�����еĽǶȶ���һ����0 ~ 2PI��
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					//���Ƕ�0 ~ 2PIת��Ϊ��ֵ0 ~ 7
					double obin = grad_ori * (bins / PI2);

					//��˹����������Ϊ����
					double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

					//����תǰ����ķ�ֵ������ת������أ�����ǰ��ֻ������ı任���൱��������ת�ͷ�����������Ϊ��ת�Ե������ط�ֵӰ�죩
					InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);

				}
			}
		}
	}

	return hist;
}

/*
  ����������ӽ��й�һ��

  @param feat ���������
*/
void NormalizeDescr(Keypoint &feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;

	//��ƽ����
	for (i = 0; i < d; i++)
	{
		cur = feat.descriptor[i];
		len_sq += cur * cur;
	}
	len_inv = 1.0 / sqrt(len_sq);

	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}

/*
  ���������������ת��Ϊ���������ʸ��

  @param hist �������������
  @param width ���򻮷�Ϊ4*4�ĸ�����
  @param bins ����������ݶȷ���ֱ��ͼ��bins
  @param feature ������������ʸ��
*/
void HistToDescriptor(double ***hist, int width, int bins, Keypoint &feature)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}
	//k = 4 * 4 * 8 = 128
	feature.descr_length = k;

	//�ض�ǰ���й�һ����ȥ�����յ�Ӱ��
	NormalizeDescr(feature);

	for (i = 0; i < k; i++)
		//���������ʸ��ĳһά��ֵ����0.2Ҫ���нض�
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;

	//�ضϺ���й�һ����������������ʸ���ļ�����
	NormalizeDescr(feature);

	/* convert floating-point descriptor to integer valued descriptor */
	for (i = 0; i < k; i++)
	{
		int_val = INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}

/*
  ������������ӣ��������������ֱ��ͼ�������������ֱ��ͼת��Ϊ���������ʸ��

  @param hist �������������
  @param width ���򻮷�Ϊ4*4�ĸ�����
  @param bins ����������ݶȷ���ֱ��ͼ��bins
  @param feature ������������ʸ��
*/
void DescriptorRepresentation(vector<Keypoint> &features, const vector<Mat> &gauss_pyr, int bins, int width)
{
	double ***hist;

	for (int i = 0; i < features.size(); i++)
	{
		hist = CalculateDescrHist(gauss_pyr[features[i].octave*(INTERVALS + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);


		HistToDescriptor(hist, width, bins, features[i]);

		for (int j = 0; j < width; j++)
		{

			for (int k = 0; k < width; k++)
			{
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}

}

/*
  �ȽϺ������������㰴�߶ȵĽ�������

  @param f1 ��һ���������ָ��
  @param f2 �ڶ����������ָ��
  @return ���f1�ĳ߶�С��f2�ĳ߶ȣ�����1�����򷵻�-1������ȷ���0
*/
bool FeatureCmp(Keypoint &f1, Keypoint &f2)
{
	return f1.scale < f2.scale;
}

//sift �㷨
void Sift(const Mat &src, vector<Keypoint> &features, double sigma, int intervals)
{
	Mat init_gray;
	//��ʼ����0���0��ͼ��
	CreateInitSmoothGray(src, init_gray, sigma);

	//�����˹������������
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;
	std::cout << "rows = " << init_gray.rows << "  cols = " << init_gray.cols << "  octaves = " << octaves << std::endl;
	std::cout << std::endl;


	std::cout << "building gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> gauss_pyr;
	//���ɸ�˹������
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);

	//write_pyr(gauss_pyr, "gausspyrmaid");
	//display_pyr(gauss_pyr, "gausspyramid");

	std::cout << "building difference of gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> dog_pyr;
	//���ɸ�˹��ֽ�����
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);

	//write_pyr(dog_pyr, "dogpyrmaid");
	//display_pyr(dog_pyr, "dogpyramid");

	std::cout << "deatecting local extrema..." << std::endl;
	vector<Keypoint> extrema;
	//���ؼ��㣬�����ɹؼ���vector
	DetectionLocalExtrema(dog_pyr, extrema, octaves, intervals);
	std::cout << "keypoints cout: " << extrema.size() << std::endl;
	std::cout << "extrema detection finished" << std::endl;
	std::cout << "please look dir gausspyramid, dogpyramid and extrema.txt" << std::endl;
	std::cout << std::endl;

	//����ؼ���vector��ÿ���ؼ�������ڵ�0���0��ĳ߶�
	CalculateScale(extrema, sigma, intervals);

	//����ؼ���vector��ÿ����ֵ��x��y��scale�����ԭͼ���x��y��scale
	HalfFeatures(extrema);

	std::cout << "orientation assignment..." << std::endl;
	//����ؼ����������
	OrientationAssignment(extrema, features, gauss_pyr);
	std::cout << "features count: " << features.size() << std::endl;
	std::cout << std::endl;

	std::cout << "generating SIFT descriptors..." << std::endl;
	std::cout << std::endl;
	//������������ӵ�ֱ��ͼ���������������ʸ��
	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);

	//�������������ʸ���ĳ߶Ƚ�������
	sort(features.begin(), features.end(), FeatureCmp);

	std::cout << "finished......" << std::endl;
	std::cout << std::endl;
}

void Sift_nointerval(const Mat &src, vector<Keypoint> &features, double sigma, int intervals)
{
	Mat init_gray;
	//��ʼ����0���0��ͼ��
	CreateInitSmoothGray(src, init_gray, sigma);

	//�����˹������������
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;
	std::cout << "rows = " << init_gray.rows << "  cols = " << init_gray.cols << "  octaves = " << octaves << std::endl;
	std::cout << std::endl;


	std::cout << "building gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> gauss_pyr;
	//���ɸ�˹������
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);

	//write_pyr(gauss_pyr, "gausspyrmaid");
	//display_pyr(gauss_pyr, "gausspyramid");

	std::cout << "building difference of gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> dog_pyr;
	//���ɸ�˹��ֽ�����
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);

	//write_pyr(dog_pyr, "dogpyrmaid");
	//display_pyr(dog_pyr, "dogpyramid");

	std::cout << "deatecting local extrema..." << std::endl;
	vector<Keypoint> extrema;
	//���ؼ��㣬�����ɹؼ���vector
	DetectionLocalExtrema_nointerval(dog_pyr, extrema, octaves, intervals);
	std::cout << "keypoints cout: " << extrema.size() << std::endl;
	std::cout << "extrema detection finished" << std::endl;
	std::cout << "please look dir gausspyramid, dogpyramid and extrema.txt" << std::endl;
	std::cout << std::endl;

	//����ؼ���vector��ÿ���ؼ�������ڵ�0���0��ĳ߶�
	CalculateScale(extrema, sigma, intervals);

	//����ؼ���vector��ÿ����ֵ��x��y��scale�����ԭͼ���x��y��scale
	HalfFeatures(extrema);

	std::cout << "orientation assignment..." << std::endl;
	//����ؼ����������
	OrientationAssignment(extrema, features, gauss_pyr);
	std::cout << "features count: " << features.size() << std::endl;
	std::cout << std::endl;

	std::cout << "generating SIFT descriptors..." << std::endl;
	std::cout << std::endl;
	//������������ӵ�ֱ��ͼ���������������ʸ��
	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);

	//�������������ʸ���ĳ߶Ƚ�������
	sort(features.begin(), features.end(), FeatureCmp);

	std::cout << "finished......" << std::endl;
	std::cout << std::endl;
}

void DrawSiftFeature(Mat &src, Keypoint &feat, cv::Scalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	cv::Point start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
	start_x = cvRound(feat.dx);
	start_y = cvRound(feat.dy);
	scl = feat.scale;
	ori = feat.ori;
	len = cvRound(scl * scale);
	hlen = cvRound(scl * hscale);
	blen = len - hlen;
	end_x = cvRound(len *  cos(ori)) + start_x;
	end_y = cvRound(len * -sin(ori)) + start_y;

	h1_x = cvRound(blen *  cos(ori + CV_PI / 18.0)) + start_x;
	h1_y = cvRound(blen * -sin(ori + CV_PI / 18.0)) + start_y;
	h2_x = cvRound(blen *  cos(ori - CV_PI / 18.0)) + start_x;
	h2_y = cvRound(blen * -sin(ori - CV_PI / 18.0)) + start_y;

	start = cv::Point(start_x, start_y);
	end = cv::Point(end_x, end_y);
	h1 = cv::Point(h1_x, h1_y);
	h2 = cv::Point(h2_x, h2_y);

	//��������ߵĳ��̴����ų߶ȵĴ�С
	line(src, start, end, color, 1, 8, 0);
	line(src, end, h1, color, 1, 8, 0);
	line(src, end, h2, color, 1, 8, 0);
	//circle(src, Point(start_x, start_y), blen, color, 1);
}

void DrawSiftFeatures(Mat &src, vector<Keypoint> &features)
{
	cv::Scalar color = CV_RGB(0, 255, 0);
	for (int i = 0; i < features.size(); i++)
	{
		DrawSiftFeature(src, features[i], color);
	}
}

void DrawKeyPoints(Mat &src, vector<Keypoint> &keypoints)
{
	int j = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{

		cv::Scalar color = CV_RGB(0, 0, 255);
		circle(src, Point(keypoints[i].dx, keypoints[i].dy), 3, color, 1);
		j++;
	}
}

void DrawKeyPointsRight(Mat &src, vector<Keypoint> &keypoints, cv::Size size)
{
	int j = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{

		cv::Scalar color = CV_RGB(0, 0, 255);
		//����ƴ��ʹ��
		circle(src, Point(keypoints[i].dx + size.width, keypoints[i].dy), 3, color);
		j++;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cv64f_to_cv8U(const Mat &src, Mat &dst)
{
	normalize(src, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
}

const char *GetFileName(const char* dir, int i)
{
	char *name = new char[50];
	//std::cout << "name: " << dir << " " << i << std::endl;
	sprintf(name, "%s\%d.jpg", dir, i);
	return name;
}

//ͨ��ת���󱣴��ͼ�񣬻�ʧ��,��imshow��ʾ����ͼ�����ܴ�
void writecv64f(const char* filename, const Mat &mat)
{
	Mat dst;
	cv64f_to_cv8U(mat, dst);
	imwrite(filename, dst);
}

void write_pyr(const vector<Mat> &pyr, const char* dir)
{
	for (int i = 0; i < pyr.size(); i++)
	{

		writecv64f(GetFileName(dir, i), pyr[i]);
	}

}

void display_pyr(const vector<Mat> &pyr, string dir)
{
	for (int i = 0; i < pyr.size(); i++)
	{
		std::cout << "name: " << dir << " " << i << std::endl;
		string name = dir + to_string(i) + ".jpg";
		std::cout << name << std::endl;
	}

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void read_features(vector<Keypoint> &features, const char* file)
{
	//�ļ����������洢�豸��ȡ���ڴ���
	ifstream in(file);
	int n = 0, dims = 0;
	//����ȡ��(>>)���ļ��ж�������
	in >> n >> dims;
	std::cout << n << " " << dims << std::endl;
	for (int i = 0; i < n; i++)
	{
		Keypoint key;
		//�����������x��y���꣬�߶�scale������ori
		in >> key.dy >> key.dx >> key.scale >> key.ori;

		//������������Ӹ�ά�ȵ�����
		for (int j = 0; j < dims; j++)
		{
			in >> key.descriptor[j];
		}
		features.push_back(key);
	}
	in.close();
}

void write_features(const vector<Keypoint> &features, const char* file)
{
	//�ļ�д�������ڴ�д��洢�豸
	ofstream dout(file);
	//�ò�����(<<)���ļ���д������
	dout << features.size() << " " << FEATURE_ELEMENT_LENGTH << endl;
	for (int i = 0; i < features.size(); i++)
	{
		//д���������x��y���꣬�߶�scale������ori
		dout << features[i].dy << " " << features[i].dx << " " << features[i].scale << " " << features[i].ori << endl;
		for (int j = 0; j < FEATURE_ELEMENT_LENGTH; j++)
		{
			//ÿ�����ݵ���ʮ���������
			if (j % 20 == 0)
				dout << endl;
			dout << features[i].descriptor[j] << " ";
		}
		dout << endl;
		dout << endl;
	}
	dout.close();
}


void testInverse3D()
{
	double H[9] = { 3,2,3, 4, 5, 6, 7, 8, 9 };
	double H_inve[9] = { 0 };
	double r[9] = { 0 };

	if (!Inverse3D(H, H_inve))
	{
		cout << "Not inverse." << endl;
		return;
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			r[3 * i + j] = 0;
			for (int k = 0; k < 3; k++)
			{
				r[3 * i + j] += H[3 * i + k] * H_inve[3 * k + j];
			}
			cout << r[3 * i + j] << " ";
		}
		cout << endl;
	}

}