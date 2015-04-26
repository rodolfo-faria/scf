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

	Tensor<double, 4> t(2,2,2,2);
	t.setConstant(2);
	//cout << "dimension of t:" << endl << tnsr(t) << "\n\n";
	cout << "Tensor: \n" << t << "\n";
	tnsr(t);
	cout << "Tensor: \n" << t << "\n";

	Matrix2f m = Matrix2f::Random();
	m = (m + m.adjoint()).eval();
	JacobiRotation<float> J;
	J.makeJacobi(m, 0, 1);
	cout << "Here is the matrix m:" << endl << m << endl;
	m.applyOnTheLeft(0, 1, J.adjoint());
	m.applyOnTheRight(0, 1, J);
	cout << "Here is the matrix J' * m * J:" << endl << m << endl;

	   EigenSolver<Matrix2f> eigensolver(m);
	   if (eigensolver.info() != Success) abort();
	   cout << "The eigenvalues of m are:\n" << eigensolver.eigenvalues() << endl;
	   cout << "Here's a matrix whose columns are eigenvectors of A \n"
	        << "corresponding to these eigenvalues:\n"
	        << eigensolver.eigenvectors() << endl;

	   Eigen::Matrix3d mmm;
	   Eigen::Matrix3d rrr;
	                   rrr <<  0.882966, -0.321461,  0.342102,
	                           0.431433,  0.842929, -0.321461,
	                          -0.185031,  0.431433,  0.882966;
	                        // replace this with any rotation matrix

	   mmm = rrr;

	   Eigen::AngleAxisd aa(rrr);    // RotationMatrix to AxisAngle
	   rrr = aa.toRotationMatrix();  // AxisAngle      to RotationMatrix

	   std::cout <<     mmm << std::endl << std::endl;
	   std::cout << rrr     << std::endl << std::endl;
	   std::cout << rrr-mmm << std::endl << std::endl;
	   std::cout << m.inverse() << std::endl << std::endl;
	   std::cout << m.inverse().cwiseSqrt() << std::endl << std::endl;



	return 0;

}


