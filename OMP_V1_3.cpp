//本版基于矩阵逆更新算法
//本程序需要配置Eigen库
//矩阵逆更新算法参考文献：《基于GPU的压缩感知重构算法的设计与实现＿张静》
//传统OMP算法参考博客：https://blog.csdn.net/jbb0523/article/details/45130793
//OMP_Version_1.3 2018.03.30_19:58 by Cooper Liu
//Questions ? Contact me : angelpoint@foxmail.com
#include"iostream"
#include <ctime>  
#include <Eigen/Eigenvalues>
using namespace std;
using namespace Eigen;

void Timer(char *title, clock_t &start, clock_t &middle, clock_t &finish, double &totaltime)//计时器
{
	printf("\n%s时间为:", title);
	finish = clock();//获取当前时间
	totaltime = (double)(finish - middle) / CLOCKS_PER_SEC;//计算中间时间
	cout << totaltime << "秒  ";
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;//计算整体时间
	cout << "totaltime:" << totaltime << "秒" << endl;
	middle = finish;
}

void swap(int *a, int i, int j)
{
	int tmp = a[i];
	a[i] = a[j];
	a[j] = tmp;
}

void randperm(int *a, int N)
{
	int i = 0;
	for (i = 0; i < N; i++)
		a[i] = i + 1;
	for (i = 0; i < N; i++)//交换a[i]和a[i]后面的随机序号
		swap(a, i, i + rand() % (N - i));
}

void main()
{
	clock_t start, middle, finish;//测时
	double totaltime;
	start = clock();//开始计时
	middle = start;

	//数据随机生成------------------------------------------------------------------
	int i = 0, j = 0;
	int M = 64, N = 256, K = 10;
	//int M = 64 * 6, N = 256 * 6, K = 10 * 6;
	//int M = 512, N = 8192, K = 64;
	int *Index_K = (int *)malloc(N*sizeof(int));
	randperm(Index_K, N);//生成随机序列，实现类似matlab中randperm(N)的功能
	Timer("随机序列生成", start, middle, finish, totaltime);

	VectorXd X = VectorXd::Zero(N);//列向量
	for (i = 0; i < K; i++)
		X(Index_K[i]) = rand() / (double)(RAND_MAX);
	Timer("原始数据X生成", start, middle, finish, totaltime);
	cout <<"原始数据X是"<< endl << X.transpose() << endl;//输出原始数据

	MatrixXd Psi = MatrixXd::Identity(N, N);//单位阵
	MatrixXd Phi = (Eigen::MatrixXd::Random(M, N)).array();//随机高斯矩阵，保证任意2K列不线性相关
	Timer("单位阵Psi和观测矩阵Phi生成", start, middle, finish, totaltime);
	//std::cout << Phi << std::endl;//输出观测矩阵Phi

	MatrixXd A(M, N);
	//A = Phi*Psi;//计算A矩阵
	A = Phi;//因为这里的Psi是单位阵，因此A就直接等于Phi
	Timer("计算出A矩阵", start, middle, finish, totaltime);

	//检验A和Phi的差距
	double miss = 0.;
	for (i = 0; i < M; i++){
		for (j = 0; j < N; j++)
			miss = A(i, j) - Phi(i, j);
	}
	cout << "miss is " << miss << endl;
	Timer("检验A和Phi的差距", start, middle, finish, totaltime);

	VectorXd y(M);
	y = Phi*X;//计算出测量值y
	Timer("计算出测量值y", start, middle, finish, totaltime);
	//cout << y;//输出测量值y

	//以下为OMP算法-----------------------------------------------------------------
	VectorXd theta = VectorXd::Zero(N);//用来存储 稀疏系数 的列向量
	MatrixXd At = MatrixXd::Zero(M, K);//用来存储 最相关列 的矩阵
	VectorXd Pos_theta = VectorXd::Zero(K);//用来存储 最相关列所在位置 的列向量
	VectorXd r_n = y;//残差
	int pos = 0;
	VectorXd theta_ls;
	MatrixXd Bminus1;//参考文献公式中的B(t-1)
	MatrixXd B;//参考文献公式中的B(t)
	MatrixXd u1, u2;//参考文献公式中u1，u2
	double d = 1.;//参考文献公式中的d
	for (i = 0; i < K; i++)
	{
		//寻找过完备冗余字典A中与残差r_n最相关的列
		VectorXd product = A.transpose()*r_n;
		VectorXd productTmp = product;
		productTmp = productTmp.cwiseAbs();//对矩阵元素取绝对值
		productTmp.maxCoeff(&pos);//返回 值最大 的位置给pos
		At.col(i) = A.col(pos);//将该列存到At中
		Pos_theta(i) = pos;//记录这个位置
		//以下为 矩阵逆更新算法
		if (i == 0)
		{
			Bminus1 = (At.leftCols(i + 1).transpose()*At.leftCols(i + 1)).inverse();
			theta_ls = Bminus1*At.leftCols(i + 1).transpose()*y;
			//cout << endl << "初始的Bminus1 is" << endl << Bminus1 << endl;
		}
		if (i > 0)
		{
			u1 = At.leftCols(i).transpose()*A.col(pos);
			u2 = Bminus1*u1;
			MatrixXd tmp = (A.col(pos).transpose()*A.col(pos) - u1.transpose()*u2);
			d = 1 / tmp(0, 0);
			MatrixXd F = Bminus1 + d*u2*u2.transpose();
			MatrixXd Btmp(i + 1, i + 1);
			Btmp.block(0, 0, i, i) = F;//左上角矩阵
			Btmp.block(0, i, i, 1) = -d*u2;//最右边一列
			Btmp.block(i, 0, 1, i) = -d*u2.transpose();//最下边一行
			Btmp(i, i) = d;//最右下角元素
			//输出过程，用来验证
			/*cout << endl << "第 " << i << " 次" << endl;
			cout << "     u1 is " << endl << u1 << endl;
			cout << "     F  is    （左上角矩阵）" << endl << F << endl;
			cout << "     -d*u2 is （最右边一列）" << endl << -d*u2 << endl;
			cout << "     -d*u2T is（最下边一行）" << endl << -d*u2.transpose() << endl;
			cout << "     d  is    （最右下角元素）" << endl << d << endl;
			cout << "     Btmp is   (上述四项合并成的矩阵）" << endl << Btmp << endl;*/
			B = Btmp;
			Bminus1 = B;//这里的B就是
			theta_ls = B*At.leftCols(i + 1).transpose()*y;
		}
		r_n = y - At.leftCols(i + 1)*theta_ls;
	}
	//恢复信号----------------------------------------------------------------------
	for (i = 0; i < K; i++)
	{
		j = (int)Pos_theta(i);
		theta(j) = theta_ls(i);
	}
	MatrixXd x_r = Psi*theta;
	Timer("恢复过程", start, middle, finish, totaltime);
	//计算信息损失------------------------------------------------------------------
	miss = 0.;
	for (i = 0; i < N; i++)
		miss = miss + abs(x_r(i) - X(i));
	cout << "info miss（信息损失） is " << miss << endl;
	Timer("计算信息损失(info miss)", start, middle, finish, totaltime);

	cout << "恢复的数据 x_r is" << endl << x_r.transpose() << endl;//输出恢复的数据
	system("pause");
}
