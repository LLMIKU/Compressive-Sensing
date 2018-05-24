//哈达玛-维纳算子恢复算法version_4.01 完成时间：2018.1.19_12:36 by Cooper Liu
//本程序需要配置Eigen库和openCV
//Questions? contact me: angelpoint@foxmail.com
#include<string.h>
#include <iostream>
//#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include<opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>
#include<time.h>
#include"kernel.cuh"
using namespace Eigen;
using namespace std;
using namespace cv;

typedef uchar * pixel;


MatrixXd im2col(int *N, int *num, MatrixXd &I, int row, int col, int b_size)
{
	int i = 0, j = 0;
	int moban_size = b_size * b_size;

	int row_addtion = (row%b_size > 0 ? 1 : 0);
	int col_addtion = (col%b_size > 0 ? 1 : 0);
	int row_part = row / b_size + row_addtion;//行数能被分为模块的数目
	int col_part = col / b_size + col_addtion;//列数能被分为模块的数目
	int row2 = moban_size;
	int col2 = row_part*col_part;
	*N = row2;
	*num = col2;

	MatrixXd X(row2, col2);
	int targetRow = 0;
	int targetCol = 0;
	for (i = 0; i < row2; i++)
	{
		for (j = 0; j < col2; j++)
		{
			targetRow = j%row_part*b_size + i%b_size;//按照规则计算目标所在行
			targetCol = j / row_part*b_size + i / b_size;//计算目标所在列
			if (targetRow >= row || targetCol >= col)//如果目标不存在
				X(i, j) = 0;
			else
				X(i, j) = I(targetRow, targetCol);//获取目标
		}
	}

	return X;
}

void col2im(/*int *N, int *num,*/ MatrixXd &I, MatrixXd &X, int row, int col, int b_size)
{
	int i = 0, j = 0;
	int moban_size = b_size * b_size;

	int row_addtion = (row%b_size > 0 ? 1 : 0);
	int col_addtion = (col%b_size > 0 ? 1 : 0);
	int row_part = row / b_size + row_addtion;//行数能被分为模块的数目
	int col_part = col / b_size + col_addtion;//列数能被分为模块的数目
	int row2 = moban_size;
	int col2 = row_part*col_part;
	//*N = row2;
	//*num = col2;

	//MatrixXd X(row2, col2);
	int targetRow = 0;
	int targetCol = 0;
	for (i = 0; i < row2; i++)
	{
		for (j = 0; j < col2; j++)
		{
			targetRow = j%row_part*b_size + i%b_size;//按照规则计算目标所在行
			targetCol = j / row_part*b_size + i / b_size;//计算目标所在列
			if (targetRow >= row || targetCol >= col)//如果目标不存在
				;
			else
				I(targetRow, targetCol) = X(i, j);//获取目标
		}
	}
}

void mean(MatrixXd &out, int row, int col, MatrixXd &in)
{
	int i = 0, j = 0;
	double tmp = 0.0;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
			tmp += in(i, j);
		tmp = tmp / col;
		for (j = 0; j < col; j++)
			out(i, j) = tmp;
		tmp = 0.0;
	}
}

int myPow(int inNum, int power)
{
	int i = 0;
	int sum = 1;
	for (i = 0; i < power; i++)
	{
		sum = sum*inNum;
	}
	return sum;
}

MatrixXd myHardmard(int n)//仅产生简单的哈达玛矩阵
{
	int i = 0;
	int j = 0;
	int k = 0;
	int findFlag = 0;
	int finalI = 0;
	int nLimit = (int)pow(2.0, 10.0);
	if (n > nLimit)
	{
		printf("无法产生那么大的哈达玛矩阵,请手动关闭本程序\n");
		system("pause");
	}
	for (i = 0; i <= 10; i++)
	{
		if (n == myPow(2, i))
		{
			findFlag = 1;
			finalI = i;
			break;
		}
	}
	if (findFlag != 1)
	{
		printf("你输入的哈达玛矩阵列数n不是2的N次方,请手动关闭本程序\n");
		system("pause");
	}
	MatrixXd H(n, n);
	int row = myPow(2, finalI);
	int col = row;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			if (i == 0 && j == 0)
				H(i, j) = 1;
			else for (k = 0; k <= 10; k++)
			{
				int tmpLimit = myPow(2, k);
				if ((i >= tmpLimit) && (j >= tmpLimit))
					H(i, j) = -H(i - tmpLimit, j - tmpLimit);

				if ((i >= tmpLimit) && (j < tmpLimit))
					H(i, j) = H(i - tmpLimit, j);

				if ((i<tmpLimit) && (j >= tmpLimit))
					H(i, j) = H(i, j - tmpLimit);
			}
		}
	}
	//%% << "生成的哈达玛矩阵是：\n" << H << endl;

	return H;
}

int checkNoRecount(int *a, int j, int count)
{
	int i = 0;
	for (i = 0; i <= count; i++)
	{
		if (j == a[i])
			return 0;
	}
	return 1;
}

void Timer(char *title, clock_t &start, clock_t &middle, clock_t &finish, double &totaltime)
{
	printf("\n%s时间为:", title);
	finish = clock();//获取当前时间
	totaltime = (double)(finish - middle) / CLOCKS_PER_SEC;//计算中间时间
	cout << totaltime << "秒  ";
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;//计算整体时间
	cout << "totaltime:" << totaltime << "秒" << endl;
	middle = finish;
}

void main()
{
	clock_t start, middle, finish;//测时
	double totaltime;
	start = clock();//开始计时
	middle = start;

	int i = 0;
	int j = 0;
	int k = 0;
	int b_size = 2;//注意不能为0
	int c = b_size*b_size / 4;
	int yt_num = 1;

	Mat img = imread("E:\\matlab\\toolbox\\images\\imdata\\testpat1.png", 0);
	//Mat img = imread("4.jpg", 0);

	int row = img.rows - img.rows%b_size;
	int col = img.cols - img.cols%b_size;
	printf("尺寸：%d*%d\n", row, col);
	namedWindow("原图");
	imshow("原图", img);
	Timer("读取原图", start, middle, finish, totaltime);

	MatrixXd mI(row, col);

	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			mI(i, j) = img.at<unsigned char>(i, j);
		}
	}
	Timer("图像转为矩阵", start, middle, finish, totaltime);

	int N = 0;//X的行数
	int num = 0;//X的列数
	MatrixXd X = im2col(&N, &num, mI, row, col, b_size);//函数返回给类的初始化不会出错
	//%% << "im2col后的矩阵：\n" << X << endl;
	Timer("im2col", start, middle, finish, totaltime);

	int N2 = 0;
	int num2 = 0;
	double *X2 = NULL;
	//X2 = im2col(&N2,&num2,)


	MatrixXd mean_x(N, num);//行平均值矩阵
	mean(mean_x, N, num, X);//求行平均值矩阵
	//%% << "行平均值矩阵：\n" << mean_x << endl;
	Timer("行平均值", start, middle, finish, totaltime);

	MatrixXd X_center(N, num);//偏差矩阵
	X_center = X - mean_x;//求偏差矩阵=原始矩阵-行平均值矩阵
	//%% << "偏差矩阵：" << endl << X_center << endl;
	Timer("求偏差矩阵", start, middle, finish, totaltime);

	MatrixXd temp(N, N);//整体偏差矩阵
	double *X_center2 = (double *)malloc(N*num*sizeof(double));
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < num; j++)
		{
			X_center2[i*num + j] = X_center(i, j);
		}
	}
	Timer("求整体偏差矩阵：矩阵传值", start, middle, finish, totaltime);
	double *temp2 = (double *)malloc(N*N*sizeof(double));
	matMul3(N, num, X_center2, temp2);
	Timer("求整体偏差矩阵：GPU计算", start, middle, finish, totaltime);
	temp = X_center*X_center.transpose();//求整体偏差矩阵=偏差差矩阵*偏差差矩阵的转置
	Timer("求整体偏差矩阵：Eigen计算", start, middle, finish, totaltime);
	double miss = 0.;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			/*double tmp = temp(i, j);
			miss = miss + tmp - temp2[i*N + j];*/
			temp(i, j) = temp2[i*N + j];
		}
	}
	printf("miss is %f\n", miss);
	//%% << "偏差矩阵和自身转置的乘积得到的矩阵：\n" << temp << endl;
	Timer("*求整体偏差矩阵：矩阵传回", start, middle, finish, totaltime);

	MatrixXd R(N, N);//平均偏差矩阵
	R = temp / num;//求平均偏差矩阵=整体偏差矩阵./num
	//%% << "平均偏差R:" << endl << R << endl;
	Timer("求平均偏差矩阵", start, middle, finish, totaltime);

	EigenSolver<MatrixXd> es(R);

	MatrixXd D = es.pseudoEigenvalueMatrix();
	MatrixXd V = es.pseudoEigenvectors();
	//%% << "The pseudo-eigenvalue matrix D is:" << endl << D << endl;
	//%% << "The pseudo-eigenvector matrix V is:" << endl << V << endl;
	//%% << "Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;
	Timer("求特征值", start, middle, finish, totaltime);

	double *lambda = (double *)malloc(N*sizeof(double));
	printf("lambad:\n");
	for (i = 0; i < N; i++)
	{
		lambda[i] = D(i, i);
		printf("%lf\n", lambda[i]);
	}
	printf("\n");
	Timer("lambda", start, middle, finish, totaltime);

	MatrixXd F0 = myHardmard(N);

	double *s = (double *)malloc(N*sizeof(double));
	memset(s, 0, N*sizeof(double));
	double tmp = 0.0;
	for (j = 0; j < N; j++)
	{
		for (i = 0; i < N; i++)
		{
			tmp = 0.0;
			for (k = 0; k < N; k++)
			{
				tmp += (double)F0(j, k) * V(k, i);
			}
			tmp = pow(tmp, 2.0);
			s[j] = s[j] + lambda[i] * tmp;
		}
	}
	printf("s:\n");
	for (j = 0; j < N; j++)
		printf("%f ", s[j]);
	printf("\n");
	Timer("求s", start, middle, finish, totaltime);

	int M = N / 4;//采样率

	int *F0maxIndex = (int *)malloc(M*sizeof(int));
	for (i = 0; i < M; i++)
		F0maxIndex[i] = -1;
	printf("index pri:\n");
	for (i = 0; i < M; i++)
		printf("%d ", F0maxIndex[i]);
	printf("\n");
	double tmpMax = 0.0;

	for (i = 0; i < M; i++)
	{
		tmpMax = -10000.0;//此处待修改
		for (j = 0; j < N; j++)
		{
			if (i == 0)
			{
				if (tmpMax < s[j])
				{
					tmpMax = s[j];
					F0maxIndex[i] = j;
				}
			}
			if (i > 0)
			{
				if ((tmpMax < s[j]) && (tmpMax <= s[F0maxIndex[i - 1]]) && checkNoRecount(F0maxIndex, j, i - 1))
				{
					tmpMax = s[j];
					F0maxIndex[i] = j;
				}
			}
		}
	}
	printf("index after:\n");
	for (i = 0; i < M; i++)
		printf("%d ", F0maxIndex[i]);
	printf("\n");
	Timer("求index", start, middle, finish, totaltime);

	MatrixXd F(M, N);
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			F(i, j) = F0(F0maxIndex[i], j);

	//%% << "F:\n" << F << endl;

	MatrixXd W;//恢复算子
	W = R*F.transpose()*(F*R*F.transpose()).inverse();
	//%% << "W:" << endl << W << endl;
	Timer("求恢复算子", start, middle, finish, totaltime);

	MatrixXd Y = F*X;//在测量矩阵F下得到的测量值
	//%% << "Y:" << endl << Y << endl;
	Timer("得到测量值Y", start, middle, finish, totaltime);

	MatrixXd recon_X = W*Y;
	//%% << "recon_X" << endl << recon_X << endl;
	Timer("求recon_X", start, middle, finish, totaltime);

	MatrixXd recon_I(row, col);
	col2im(recon_I, recon_X, row, col, b_size);
	//%% << "col2im后的矩阵recon_I：\n" << recon_I << endl;
	printf("\n恢复完成\n");
	Timer("col2im", start, middle, finish, totaltime);



	Mat img3 = Mat(row, col, CV_8UC1);
	unsigned char *ptmp = NULL;
	for (i = 0; i < row; i++)
	{
		ptmp = img3.ptr<unsigned char>(i);
		for (j = 0; j < col; j++)
		{
			if (recon_I(i, j) < 0.0)
				ptmp[j] = 0;
			else if (recon_I(i, j) > 255.0)
				ptmp[j] = 254;
			else
				ptmp[j] = recon_I(i, j);
		}
	}
	Timer("矩阵传给图像", start, middle, finish, totaltime);

	namedWindow("恢复图");
	imshow("恢复图", img3);
	Timer("此程序", start, middle, finish, totaltime);

	waitKey();

}
