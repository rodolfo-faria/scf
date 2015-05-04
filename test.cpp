/*
 * test.cpp
 *
 *  Created on: Apr 25, 2015
 *      Author: belzebu
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>
#include <vector>
#include "Read.h"

using namespace Eigen;
using namespace std;


double func(Ref<MatrixXd> x)
//double func(MatrixXd& a)
{
	x << 1, 2, 3, 4;
	double tr;
	tr = x.trace();
	return tr;
}


void tnsr(Tensor<double, 4>& var)
{
	//cout << "dimension of t:" << endl << var.NumDimensions << "\n\n";
	var.setConstant(1.0f);
	var(1,0,0,0) = 56;
}


int main(int argc, char* argv[])
{

	Matrix2d x = Matrix2d::Zero();
	cout << "matrix x:" << endl << x << endl << endl;
	double res = func(x);
	cout << "trace is " << res << endl;

	/*
	Tensor<double, 4> t(2,2,2,2);
	t.setConstant(2);
	//cout << "dimension of t:" << endl << tnsr(t) << "\n\n";
	cout << "Tensor: \n" << t << "\n";
	tnsr(t);
	cout << "Tensor: \n" << t << "\n";
	*/

	chrono::time_point<chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();

	Read parse;

	MatrixXd Coord = parse.readMatrix(argv[1]);


	//cout << parse.readMatrix(argv[1]) << endl;
	cout << Coord << endl;

	//MatrixXd Mat =  Map< MatrixXd > (data, dim/4, dim);
	//cout << "data:\n" << Mat << endl;

	//cout << Coord << endl;

	end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end-start;

	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return 0;

}


