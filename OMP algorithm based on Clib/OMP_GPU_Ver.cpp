//本版目的：恢复稀疏信号
//请先使用对应的Matlab程序DCT.m生成原始数据，并更改对应的测量值个数M，信号维度N，稀疏度K
//OMP_GPU_Version_1.0 2018.04.15_17:47 by Cooper Liu
//Questions ? Contact me : angelpoint@foxmail.com
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "common.h"
#include "cmemory.h"
#include "container.h"
#include "math_functions.h"
#include "unistd.h"
#include <stdlib.h>
#include <cstdlib>
#include "solver.hpp"	
#include <ostream>
#include<opencv2/opencv.hpp>
using namespace cv;

template <typename T>
void getLeftCols(Container<T> &des, Container<T> &src)//把src的左边的列赋给des，列数为des的宽度
{
	int n = des.width();
	if (des.height() != src.height())
	{
		printf("err:getLeftCols(): des.height!=src.height\n");
		system("pause");
	}
	if (des.width() > src.width())
	{
		printf("err:getLeftCols(): des.width > src.width\n");
		system("pause");
	}
	T* src_ptr = src.mutable_cpu_data();
	T* des_ptr = des.mutable_cpu_data();

	for (int i = 0; i < des.height(); i++){
		for (int j = 0; j < des.width(); j++){
			des_ptr[i*des.width() + j] = src_ptr[i*src.width() + j];
		}
	}
}

template <typename T>
void getLeftCol(Container<T> &des, Container<T> &src, int pos)//获取src的第pos列
{
	int n = des.width();
	if (des.height() != src.height())
	{
		printf("err:getLeftCols(): des.height!=src.height\n");
		system("pause");
	}
	if (des.width() > src.width())
	{
		printf("err:getLeftCols(): des.width > src.width\n");
		system("pause");
	}
	T* src_ptr = src.mutable_cpu_data();
	T* des_ptr = des.mutable_cpu_data();

	for (int i = 0; i < des.height(); i++){
		des_ptr[i] = src_ptr[i*src.width() + pos];
	}
}

template <typename T>
void matrxiCombine(Container<T> &B, Container<T> &F, Container<T> &u2, T d)//合并矩阵
{
	T* B_ptr = B.mutable_cpu_data();
	T* F_ptr = F.mutable_cpu_data();
	T* u2_ptr = u2.mutable_cpu_data();
	int m = 0, n = 0;
	for (int i = 0; i < B.height(); i++)
	{
		for (int j = 0; j < B.width(); j++)
		{
			if (i != F.height() && j != F.width())
				B_ptr[i*B.width() + j] = F_ptr[i*F.width() + j];
			else if (i == F.height() && j != F.width())
				B_ptr[i*B.width() + j] = -d*u2_ptr[m++];
			else if (i != F.height() && j == F.width())
				B_ptr[i*B.width() + j] = -d*u2_ptr[n++];
			else if (i == F.height() && j == F.width())
				B_ptr[i*B.width() + j] = d;
		}
	}
}

// shi li hua
template class Container<double>;
template class Container<double>;
template class Container<int>;
template class Container<unsigned int>;

void TimerCPU(char *title, clock_t &start, clock_t &middle, clock_t &finish, double &totaltime)//计时器
{
	printf("\n%s用时为:", title);
	finish = clock();//获取当前时间
	totaltime = (double)(finish - middle) / CLOCKS_PER_SEC;//计算中间时间
	std::cout << totaltime << "秒  ";
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;//计算整体时间
	std::cout << "totaltime:" << totaltime << "秒" << endl;
	middle = finish;
}

int main(){
	clock_t start, middle, finish;//测时
	double totaltime;
	start = clock();//开始计时
	middle = start;

	int i = 0, j = 0;
	int m = 0, n = 0, k = 0;
	int M = 318, N = 1024, K = 49;
	Container<double> A(1, M, N);
	Container<double> x_org(1, N, 1);
	Container<double> y(1, M, 1);
	Container<double> x(1, N, 1);
	// read data from file
	const char* file_A = "F:\\csDatabase\\A_dct_omp";
	const char* file_y = "F:\\csDatabase\\Y_dct_omp";
	const char* file_x = "F:\\csDatabase\\x_dct_omp";
	//const char* file_rx = "rx_omp";
	A.read_from_text(file_A);
	y.read_from_text(file_y);
	x_org.read_from_text(file_x);
	/*printf("\nx_org:");
	x_org.Log_data();
	printf("\ny:");
	y.Log_data();*/
	printf("\nx_org:");
	x_org.Log_data();
	printf("\n");
	TimerCPU("A,Y,X从文件读取", start, middle, finish, totaltime);

	Container<double> theta(1, N, 1);
	Container<double> At(1, M, K);
	Container<double> Pos_theta(1, K, 1);
	Container<double> r_n(1, M, 1);
	Container<double> theta_ls_final(1, K, 1);

	int pos = 0;

	double* A_ptr = A.mutable_cpu_data();
	double* y_ptr = y.mutable_cpu_data();
	double* x_org_ptr = x_org.mutable_cpu_data();
	double* x_ptr = x.mutable_cpu_data();
	double* theta_ptr = theta.mutable_cpu_data();
	double* At_ptr = At.mutable_cpu_data();
	double* Pos_theta_ptr = Pos_theta.mutable_cpu_data();
	double* r_n_ptr = r_n.mutable_cpu_data();

	r_n.share_data(y);//初始化r_n：注意我改写了原来的share_data()函数
	printf("\nr_n:"); r_n.Log_data(); printf("\n");
	const double *d_A = A.gpu_data();

	double *d_At = At.mutable_gpu_data();
	Container<double> Bminus(1, 1, 1);

	char filepath[256] = "F:\\csDatabase\\omp_gpu.txt";
	FILE *fp = fopen(filepath, "w");
	if (fp == NULL)
	{
		printf("create rx err\n");
		system("pause");
		return 0;
	}

	start = clock();//重新开始计时
	middle = start;
	//TimerCPU("计算开始", start, middle, finish, totaltime);
	for (i = 0; i < K; i++)
	{
		printf("第%d次------------------------------------------------------\n", i);
		TimerCPU("计算开始", start, middle, finish, totaltime);
		Container<double> product(1, N, 1);
		double* product_ptr = product.mutable_cpu_data();
		double *d_product = product.mutable_gpu_data();
		const double *d_r_n = r_n.gpu_data();
		c_gpu_gemm<double>(CblasTrans,
			CblasNoTrans, N, 1, M,
			1.0f, d_A, d_r_n, 0.0f,
			d_product);

		int pos = 0;
		cublasIdamax(Csingleton::cublas_handle(), N, (const double *)d_product, 1, &pos);
		pos--;
		//cout << "pos is " << pos << endl;
		At_ptr = At.mutable_cpu_data();//这句话很重要，尽管之前有
		int posTmp = pos;
		for (j = i; j < M*K; j += K)
		{
			At_ptr[j] = A_ptr[posTmp];
			posTmp += N;
		}//保存该列		
		d_At = At.mutable_gpu_data();//这句话很重要，尽管之前有

		Pos_theta_ptr[i] = pos;//记录位置

		Container<double> B(1, i + 1, i + 1);
		Container<double> theta_ls(1, i + 1, 1);

		if (i == 0)//确定Bminus
		{
			Container<double> AtTmp(1, K, K);
			double *d_AtTmp = AtTmp.mutable_gpu_data();

			c_gpu_gemm<double>(CblasTrans,
				CblasNoTrans, K, K, M,
				1.0f, d_At, d_At, 0.0f,
				d_AtTmp);
			AtTmp.mutable_cpu_data();

			double *Bminus_ptr = Bminus.mutable_cpu_data();
			double *AtTmp_ptr = AtTmp.mutable_cpu_data();
			Bminus_ptr[i] = 1 / AtTmp_ptr[i];
			Bminus.mutable_gpu_data();
			Bminus.mutable_cpu_data();

			Container<double> AtLeftCols(1, M, i + 1);
			getLeftCols(AtLeftCols, At);
			AtLeftCols.mutable_gpu_data();//这句话很重要，尽管之前有
			AtLeftCols.mutable_cpu_data();

			double *d_AtLeftCols = AtLeftCols.mutable_gpu_data();
			double *d_Bminus = Bminus.mutable_gpu_data();

			Container<double> mid(1, 1, M);
			double *d_mid = mid.mutable_gpu_data();
			c_gpu_gemm<double>(CblasNoTrans,
				CblasTrans, i + 1, M, i + 1,
				1.0f, d_Bminus, d_AtLeftCols, 0.0f,
				d_mid);

			double *d_y = y.mutable_gpu_data();
			double *d_theta_ls = theta_ls.mutable_gpu_data();
			c_gpu_gemm<double>(CblasNoTrans,
				CblasNoTrans, 1, 1, M,
				1.0f, d_mid, d_y, 0.0f,
				d_theta_ls);
			theta_ls.mutable_cpu_data();
		}
		if (i > 0)
		{
			//printf("第%d次----------------------------------------------------------------\n", i);
			Container<double> u1(1, i, 1);
			Container<double> u2(1, i, 1);
			Container<double> F(1, i, i);//左上角矩阵
			//计算u1
			Container<double> AtLeftCols(1, M, i);
			Container<double> AColPos(1, M, 1);
			getLeftCols(AtLeftCols, At);//CPU端获取At矩阵左边的列
			getLeftCol(AColPos, A, pos);//CPU端获取A的某一列
			double *d_u1 = u1.mutable_gpu_data();
			double *d_AtLeftCols = AtLeftCols.mutable_gpu_data();
			double *d_AColPos = AColPos.mutable_gpu_data();
			c_gpu_gemm<double>(CblasTrans, CblasNoTrans, i, 1, M, 1.0f, d_AtLeftCols, d_AColPos, 0.0f, d_u1);
			u1.mutable_cpu_data();

			//计算u2
			Container<double> Btmp(1, i + 1, i + 1);
			double *d_Bminus = Bminus.mutable_gpu_data();
			d_u1 = u1.mutable_gpu_data();
			double *d_u2 = u2.mutable_gpu_data();
			c_gpu_gemm<double>(CblasNoTrans, CblasNoTrans,
				Bminus.height(), 1, Bminus.width(),
				1.0f, d_Bminus, d_u1, 0.0f, d_u2);
			u2.mutable_cpu_data();

			//计算tmp
			double tmp;
			Container<double> tmp_1(1, AColPos.width(), AColPos.height());
			double *d_tmp_1 = tmp_1.mutable_gpu_data();
			c_gpu_gemm<double>(CblasTrans, CblasNoTrans,
				AColPos.width(), AColPos.width(), AColPos.height(),
				1.0f, d_AColPos, d_AColPos, 0.0f, d_tmp_1);

			Container<double> tmp_2(1, u1.width(), u2.width());
			double *d_tmp_2 = tmp_2.mutable_gpu_data();
			d_u1 = u1.mutable_gpu_data();
			d_u2 = u2.mutable_gpu_data();
			c_gpu_gemm<double>(CblasTrans, CblasNoTrans,
				u1.width(), u2.width(), u1.height(),
				1.0f, d_u1, d_u2, 0.0f, d_tmp_2);

			double *tmp_1_ptr = tmp_1.mutable_cpu_data();
			double *tmp_2_ptr = tmp_2.mutable_cpu_data();
			tmp = tmp_1_ptr[0] - tmp_2_ptr[0];
			double d = 1 / tmp;

			//计算F
			double *d_F = F.mutable_gpu_data();
			c_gpu_gemm<double>(CblasNoTrans, CblasTrans,
				u2.height(), u2.height(), u2.width(),
				d, d_u2, d_u2, 0.0f, d_F);
			d_Bminus = Bminus.mutable_gpu_data();
			c_gpu_axpy<double>(F.size(), 1., d_Bminus, d_F);
			F.mutable_cpu_data();

			//合并矩阵
			matrxiCombine(B, F, u2, d);
			Bminus.share_data(B);

			//更新theta_ls
			Container<double> AtLeftCols2(1, M, i + 1);
			Container<double> theta_ls_1(1, i + 1, M);
			getLeftCols(AtLeftCols2, At);

			double *d_B = B.mutable_gpu_data();
			double *d_AtLeftCols2 = AtLeftCols2.mutable_gpu_data();
			double *d_theta_ls_1 = theta_ls_1.mutable_gpu_data();
			c_gpu_gemm<double>(CblasNoTrans, CblasTrans,
				B.height(), AtLeftCols2.height(), B.width(),
				1.0, d_B, d_AtLeftCols2, 0.0, d_theta_ls_1);

			double *d_theta_ls = theta_ls.mutable_gpu_data();
			double *d_y = y.mutable_gpu_data();
			d_theta_ls_1 = theta_ls_1.mutable_gpu_data();
			c_gpu_gemm<double>(CblasNoTrans, CblasNoTrans,
				theta_ls_1.height(), y.width(), theta_ls_1.width(),
				1.0, d_theta_ls_1, d_y, 0.0f, d_theta_ls);

			theta_ls.mutable_cpu_data();
			theta_ls_final.share_data(theta_ls);
		}
		//更新残值r_n
		Container<double> AtLeftCols(1, M, i + 1);
		getLeftCols(AtLeftCols, At);
		double *d_AtLeftCols = AtLeftCols.mutable_gpu_data();
		double *d_theta_ls = theta_ls.mutable_gpu_data();
		Container<double> yTmp(1, M, 1);
		double *d_yTmp = yTmp.mutable_gpu_data();
		//计算yTmp = -(At.leftCols(i + 1)*theta_ls)
		c_gpu_gemv<double>(CblasNoTrans, M, i + 1, -1., d_AtLeftCols, d_theta_ls, 0., d_yTmp);
		yTmp.mutable_cpu_data();
		//令r_n=yTmp
		r_n.share_data(yTmp);
		//计算r_n = y + (yTmp) = y + (-At.leftCols(i + 1)*theta_ls)
		const double *d_y = y.mutable_gpu_data();
		double *d_r_n2 = r_n.mutable_gpu_data();
		c_gpu_axpy<double>(y.height(), 1., d_y, d_r_n2);
		r_n.mutable_cpu_data();
		//计算r_n = y - At.leftCols(i + 1)*theta_ls;
		finish = clock();//获取当前时间
		double t = (double)(finish - middle) * 1000 / CLOCKS_PER_SEC;
		fprintf(fp, "%f ", t);
		TimerCPU("计算结束", start, middle, finish, totaltime);
	}
	//TimerCPU("计算结束", start, middle, finish, totaltime);
	fclose(fp);

	printf("theta_ls_final.Log_data() is\n"); theta_ls_final.Log_data(); printf("\n");
	theta_ptr = theta.mutable_cpu_data();
	Pos_theta_ptr = Pos_theta.mutable_cpu_data();
	double *theta_ls_ptr = theta_ls_final.mutable_cpu_data();
	for (i = 0; i < K; i++)
	{
		j = (int)Pos_theta_ptr[i];
		theta_ptr[j] = theta_ls_ptr[i];
	}
	//printf("theta.Log_data() is\n"); theta.Log_data(); printf("\n");
	TimerCPU("恢复过程", start, middle, finish, totaltime);

	system("pause");
	return 0;
}
