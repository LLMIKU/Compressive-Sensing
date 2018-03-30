//本程序需要配置Eigen库
//OMP算法参考博客：https://blog.csdn.net/jbb0523/article/details/45130793
//程序改写自博客中的Matlab程序
//OMP_Version_1.0 2018.03.28_16:35 by Cooper Liu
//Questions ? Contact me : angelpoint@foxmail.com
#include"iostream"
#include <ctime>  
#include <Eigen/Eigenvalues>
using namespace std;
using namespace Eigen;

void swap(int *a, int i, int j)
{
	int tmp = a[i];
	a[i] = a[j];
	a[j] = tmp;
}

void randperm(int *a,int N)
{
	int i = 0;
	for (i = 0; i < N; i++)
		a[i] = i + 1;
	for (i = 0; i < N; i++)//交换a[i]和a[i]后面的随机序号
		swap(a, i, i + rand() % (N - i));
}

void main()
{
	int i = 0, j = 0;
	int M = 64, N = 256, K = 10;
	//int M = 64*5, N = 256*5, K = 10*5;
	int *Index_K = (int *)malloc(N*sizeof(int));
	randperm(Index_K, N);
	
	VectorXd X = VectorXd::Zero(N);//列向量
	for (i = 0; i < K; i++)
		X(Index_K[i]) = rand() / (double)(RAND_MAX);

	cout << "原始信号" << endl << X.transpose() << endl;
		
	MatrixXd Psi = MatrixXd::Identity(N, N);//单位阵
	MatrixXd Phi = (Eigen::MatrixXd::Random(M, N)).array();//随机高斯矩阵，保证任意2K列不线性相关
	//std::cout << Phi << std::endl;
	MatrixXd A(M, N);
	A = Phi*Psi;//A矩阵
	double miss = 0.;//检验A和Phi的差距
	for (i = 0; i < M; i++){
		for (j = 0; j < N; j++)
			miss = A(i, j) - Phi(i, j);
	}
	cout << "A和Phi miss is " << miss << endl;
	VectorXd y(M);
	y = Phi*X;//测量值
	//cout << y;

	//以下为OMP算法
	VectorXd theta = VectorXd::Zero(N);
	MatrixXd At = MatrixXd::Zero(M, K);
	VectorXd Pos_theta = VectorXd::Zero(K);
	VectorXd r_n = y;
	int pos = 0;
	VectorXd theta_ls;

	for (i = 0; i < K; i++)
	{
		VectorXd product = A.transpose()*r_n;
		VectorXd productTmp = product;
		productTmp = productTmp.cwiseAbs();
		productTmp.maxCoeff(&pos);
		for (j = 0; j < M; j++)
			At(j, i) = A(j, pos);
		Pos_theta(i) = pos;
		theta_ls = (At.leftCols(i+1).transpose()*At.leftCols(i+1)).inverse()*At.leftCols(i+1).transpose()*y;
		r_n = y - At.leftCols(i+1)*theta_ls;
	}
	for (i = 0; i < K; i++)
	{
		j = (int)Pos_theta(i);
		theta(j) = theta_ls(i);
	}
	
	MatrixXd x_r = Psi*theta;
	cout << "恢复的信号x_r is" << endl;
	cout << x_r.transpose() << endl;

	miss = 0.;
	for (i = 0; i < N; i++)
		miss = miss + abs(x_r(i) - X(i));
	cout << "info miss（信息损失） is " << miss << endl;
	system("pause");
}
