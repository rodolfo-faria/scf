/*
 * scf.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: belzebu
 */
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;

	//Global Variables
    //-------------------------------------------------------------
   int sto_ng = 3;
   double r_AB = 1.4632e0;
   double orb_exp_1 = 2.0925e0;
   double orb_exp_2 = 1.24e0;
   int z_A = 2;
   int z_B = 1;


   void Int1e(const Ref<const VectorXd>& scaled_exp_1, const Ref<const VectorXd>& scaled_exp_2,
		   const Ref<const VectorXd>& normalized_coef_1, const Ref<const VectorXd>& normalized_coef_2,
		    Ref<MatrixXd> S, Ref<MatrixXd> T, Ref<MatrixXd> V_A, Ref<MatrixXd> V_B);

   void Int2e(const Ref<const VectorXd>& scaled_exp_1, const Ref<const VectorXd>& scaled_exp_2,
   		   const Ref<const VectorXd>& normalized_coef_1, const Ref<const VectorXd>& normalized_coef_2,
   		   Tensor<double, 4>& TT_ijkl);

   //overlap integral for un-normalized primitives
   double OverlapInt( double A, double B, double r_AB2);
   //Calculates kinetic energy integrals for un-normalized primitives
   double KineticInt( double A, double B, double r_AB2);
   //Calculates un-normalized nuclear attraction integrals
   double PotentialInt( double A, double B, double r_AB2, double r_XP2, int z_X);
   //Calculates the F function only for s-type orbitals
   double F0( double arg );
   //Error function
   double ERF( double arg );
   //Calculate two electrons integrals
   double TwoEIntegral(double A, double B, double C, double D, double r_AB2, double r_CD2, double r_PQ2);

   void SCF(Ref<MatrixXd> H, Ref<MatrixXd> S, Tensor<double, 4>& TT_ijkl);


int main(int argc, char* argv[])
{

	// basis set coefficients
	VectorXd coefs(sto_ng);

	if(sto_ng == 1) coefs <<      1.0;
	if(sto_ng == 2) coefs << 0.678914, 0.430129;
	if(sto_ng == 3) coefs << 0.444635, 0.535328, 0.154329;


	//basis set exponents
	VectorXd exps(sto_ng);

	if(sto_ng == 1) exps << 0.270950;
	if(sto_ng == 2) exps << 0.151623, 0.851819;
	if(sto_ng == 3) exps << 0.109818, 0.405771, 2.227660;

	/*
	 scale the coefficients of primitive gaussians
	 and include normalization in contraction
	 coefficients
	*/

	VectorXd scaled_exp_1(sto_ng), scaled_exp_2(sto_ng);
	VectorXd normalized_coef_1(sto_ng), normalized_coef_2(sto_ng);

	scaled_exp_1 = exps * pow(orb_exp_1,2);
	scaled_exp_2 = exps * pow(orb_exp_2,2);

	normalized_coef_1 =  pow(2.0e0/M_PI, 0.75e0) * scaled_exp_1.array().pow(0.75) * coefs.array();
	normalized_coef_2 =  pow(2.0e0/M_PI, 0.75e0) * scaled_exp_2.array().pow(0.75) * coefs.array();


	Matrix2d S, T, V_A, V_B;

	//--------------------
	Int1e(scaled_exp_1, scaled_exp_2, normalized_coef_1, normalized_coef_2, S, T, V_A, V_B);
	//--------------------

	cout << "S: \n" << S << endl << endl;
	cout << "T: \n" << T << endl << endl;
	cout << "V_A: \n" << V_A << endl << endl;
	cout << "V_B: \n" << V_B << endl << endl;

	Tensor<double, 4> TT_ijkl(2,2,2,2);

	cout << "Dims " << TT_ijkl.NumDimensions << "\n";

	//--------------------
	Int2e(scaled_exp_1, scaled_exp_2, normalized_coef_1, normalized_coef_2, TT_ijkl);
	//--------------------


	Matrix2d H;

	H = T + V_A + V_B;

	SCF(H, S, TT_ijkl);


	return 0;

}



void Int1e(const Ref<const VectorXd>& scaled_exp_1, const Ref<const VectorXd>& scaled_exp_2,
		   const Ref<const VectorXd>& normalized_coef_1,const Ref<const VectorXd>& normalized_coef_2,
		    Ref<MatrixXd> S, Ref<MatrixXd> T, Ref<MatrixXd> V_A, Ref<MatrixXd> V_B)
{
	double r_AP; 		     //distante between atom A and point P
	double r_AP2, r_BP2;     //squared distante between atom A(B) and point P
	double r_AB2; 		     //squared distance between atoms A and B
	double S12; 		     //overlap matrix S_{12}
	double T11, T12, T22;  	 //kinetic energy matrix
	double V11A, V12A, V22A,
		   V11B, V12B, V22B; // potential energy matrix

	r_AB2 = pow(r_AB,2);

	// initialize variables
	S12 = 0.0e0;
	r_AP2 = r_BP2 = 0.0e0;
	T11 = T12 = T22 = 0.0e0;
	V11A = V12A = V22A = V11B = V12B = V22B = 0.0e0;

	for(int i = 0; i < sto_ng; i++){
		for(int j = 0; j < sto_ng; j++){

			r_AP  = scaled_exp_2(j) * r_AB / (scaled_exp_1(i) + scaled_exp_2(j));
			r_AP2 = pow(r_AP,2);
			r_BP2 = pow(r_AB - r_AP,2);

			S12 += (normalized_coef_1(i) * normalized_coef_2(j)) * OverlapInt(scaled_exp_1(i),scaled_exp_2(j),r_AB2);

			T11 += (normalized_coef_1(i) * normalized_coef_1(j)) * KineticInt(scaled_exp_1(i),scaled_exp_1(j),0.0e0);
			T12 += (normalized_coef_1(i) * normalized_coef_2(j)) * KineticInt(scaled_exp_1(i),scaled_exp_2(j),r_AB2);
			T22 += (normalized_coef_2(i) * normalized_coef_2(j)) * KineticInt(scaled_exp_2(i),scaled_exp_2(j),0.0e0);

			V11A += (normalized_coef_1(i) * normalized_coef_1(j)) * PotentialInt(scaled_exp_1(i),scaled_exp_1(j),0.0e0,0.0e0,z_A);
			V12A += (normalized_coef_1(i) * normalized_coef_2(j)) * PotentialInt(scaled_exp_1(i),scaled_exp_2(j),r_AB2,r_AP2,z_A);
			V22A += (normalized_coef_2(i) * normalized_coef_2(j)) * PotentialInt(scaled_exp_2(i),scaled_exp_2(j),0.0e0,r_AB2,z_A);

			V11B += (normalized_coef_1(i) * normalized_coef_1(j)) * PotentialInt(scaled_exp_1(i),scaled_exp_1(j),0.0e0,r_AB2,z_B);
			V12B += (normalized_coef_1(i) * normalized_coef_2(j)) * PotentialInt(scaled_exp_1(i),scaled_exp_2(j),r_AB2,r_BP2,z_B);
			V22B += (normalized_coef_2(i) * normalized_coef_2(j)) * PotentialInt(scaled_exp_2(i),scaled_exp_2(j),0.0e0,0.0e0,z_B);

		}
	}

	cout << "S12: " << S12 << endl;

	S << 1.0, S12,
		 S12, 1.0;

	T << T11, T12,
		 T12, T22;

	V_A << V11A, V12A,
		 V12A, V22A;

	V_B << V11B, V12B,
		   V12B, V22B;




}



void Int2e(const Ref<const VectorXd>& scaled_exp_1, const Ref<const VectorXd>& scaled_exp_2,
		   const Ref<const VectorXd>& normalized_coef_1,const Ref<const VectorXd>& normalized_coef_2,
		   Tensor<double, 4>& TT_ijkl)
{
    //distances
    double r_AP, r_BP, r_AQ, r_BQ, r_PQ;
    double r_AP2, r_BP2, r_AQ2, r_BQ2, r_PQ2;
    double r_AB2 = pow(r_AB, 2);

    //potentials
    double V1111, V2111, V2121, V2211, V2221, V2222;




    for(int i = 0; i < sto_ng; i++) {
    	for(int j = 0; j < sto_ng; j++) {
        	for(int k = 0; k < sto_ng; k++) {
            	for(int l = 0; l < sto_ng; l++) {

                r_AP = scaled_exp_2(i)*r_AB/(scaled_exp_2(i)+scaled_exp_1(j));
                r_BP = r_AB - r_AP;
                r_AQ = scaled_exp_2(k)*r_AB/(scaled_exp_2(k)+scaled_exp_1(l));
                r_BQ = r_AB - r_AQ;
                r_PQ = r_AP - r_AQ;

                r_AP2 = pow(r_AP,2);
                r_BP2 = pow(r_BP,2);
                r_AQ2 = pow(r_AQ,2);
                r_BQ2 = pow(r_BQ,2);
                r_PQ2 = pow(r_PQ,2);

                V1111 += normalized_coef_1(i) * normalized_coef_1(j) * normalized_coef_1(k) * normalized_coef_1(l) * TwoEIntegral(scaled_exp_1(i),scaled_exp_1(j),scaled_exp_1(k),scaled_exp_1(l),0.0e0,0.0e0,0.0e0);
                V2111 += normalized_coef_2(i) * normalized_coef_1(j) * normalized_coef_1(k) * normalized_coef_1(l) * TwoEIntegral(scaled_exp_2(i),scaled_exp_1(j),scaled_exp_1(k),scaled_exp_1(l),r_AB2,0.0e0,r_AP2);
                V2121 += normalized_coef_2(i) * normalized_coef_1(j) * normalized_coef_2(k) * normalized_coef_1(l) * TwoEIntegral(scaled_exp_2(i),scaled_exp_1(j),scaled_exp_2(k),scaled_exp_1(l),r_AB2,r_AB2,r_PQ2);
                V2211 += normalized_coef_2(i) * normalized_coef_2(j) * normalized_coef_1(k) * normalized_coef_1(l) * TwoEIntegral(scaled_exp_2(i),scaled_exp_2(j),scaled_exp_1(k),scaled_exp_1(l),0.0e0,0.0e0,r_AB2);
                V2221 += normalized_coef_2(i) * normalized_coef_2(j) * normalized_coef_2(k) * normalized_coef_1(l) * TwoEIntegral(scaled_exp_2(i),scaled_exp_2(j),scaled_exp_2(k),scaled_exp_1(l),0.0e0,r_AB2,r_BQ2);
                V2222 += normalized_coef_2(i) * normalized_coef_2(j) * normalized_coef_2(k) * normalized_coef_2(l) * TwoEIntegral(scaled_exp_2(i),scaled_exp_2(j),scaled_exp_2(k),scaled_exp_2(l),0.0e0,0.0e0,0.0e0);
                                    }
                            }
                    }
            }

    TT_ijkl(0,0,0,0) = V1111;
    TT_ijkl(1,0,0,0) = TT_ijkl(0,1,0,0) = TT_ijkl(0,0,1,0) = TT_ijkl(0,0,0,1) = V2111;
    TT_ijkl(1,0,1,0) = TT_ijkl(0,1,1,0) = TT_ijkl(1,0,0,1) = TT_ijkl(0,1,0,1) = V2121;
    TT_ijkl(1,1,0,0) = TT_ijkl(0,0,1,1) = V2211;
    TT_ijkl(1,1,1,1) = V2222;

}



double OverlapInt( double A, double B, double r_AB2)
{
	cout << "A = " << A << endl;
	return pow(M_PI/(A + B),1.5e0)*exp(- A * B * r_AB2 / (A + B));
}



double KineticInt( double A, double B, double r_AB2)
{
	return (A*B / (A+B) * (3.0e0 - 2.0e0 * A*B * r_AB2/(A+B)) * pow((M_PI/(A+B)), 1.5e0) * exp(-A*B*r_AB2/(A+B)));
}



double PotentialInt( double A, double B, double r_AB2, double r_XP2, int z_X)
{
	double V;
	V = 2.0e0 * M_PI / (A + B) * F0((A + B) * r_XP2) * exp(-A * B * r_AB2/(A + B));
	V = -V * z_X;
	return V;
}



double F0( double arg )
{
	if (arg < 1.0e-6)
		return 1.0e0 - arg/3.0e0;
	else
		return (sqrt(M_PI/arg)*ERF(sqrt(arg))/2);
}



double ERF( double arg )
{
	double A[5];

	A[0] = 0.254829592e0;
	A[1] = -0.284496736e0;
	A[2] = 1.421413741e0;
	A[3] = -1.453152027e0;
	A[4] = 1.01405429e0;

	double P = 0.3275911e0;

	double T = 1.0e0 / (1.0e0 + P*arg);
	double TN = T;
	double POLY = A[0] * TN;

	for(int i = 1; i < 5; i++  ){

		TN = TN * T;
		POLY += A[i] * TN;
	}

	return 1.0e0 - POLY * exp(-arg*arg);
}



double TwoEIntegral(double A, double B, double C, double D, double r_AB2, double r_CD2, double r_PQ2)
{
        return (2.0e0*pow(M_PI, 2.5e0)  / ( (A+B)*(C+D)*sqrt(A+B+C+D) ) * F0( (A+B) * (C+D) * r_PQ2 / (A+B+C+D) ) * exp((-A*B*r_AB2)/(A+B) - C * D * r_CD2 / (C+D)));
}



void SCF(Ref<MatrixXd> H, Ref<MatrixXd> S, Tensor<double, 4>& TT_ijkl)
{

	const int maxiter = 15; // maximum number of iteration
	int iter; // iteration number
	double delta = 0.0e0; // energy difference between cycles
	double electronic_energy = 0.0;
	int nocc = 1;

	// In a more general case, these matrices
	// 	have size of the # basis functions
	Matrix2d F, F_p, G, C, C_p, P, E, P_old;

	/* Calculate the symmetric transformation matrix
	 	 by diagonalizing the overlap matrix S
	 	 Note that .array() is needed to calculate the inverse
	 	 of the coefficients and not the inverse matrix
	---------------*/
	SelfAdjointEigenSolver<MatrixXd> s_matrix(S);
	VectorXd eigen_values = s_matrix.eigenvalues().cwiseSqrt().array().inverse();

	MatrixXd Lambda = eigen_values.asDiagonal();
	MatrixXd L_s = s_matrix.eigenvectors();

	MatrixXd X = L_s * Lambda * L_s.transpose();
	//---------------

	Matrix2d Error_Matrix = Matrix2d::Zero();

	iter = 0;

	while( iter < maxiter )
	{

		// Form G by accessing the tensor which
		// holds the 2e integrals
        for( int i = 0; i < 2; i++){
        	for( int j = 0; j < 2; j++){
                G(i,j) = 0.0e0;
                for( int k = 0; k < 2; k++){
                	for( int l = 0; l < 2; l++){

                	G(i,j) += P(k,l) * ( TT_ijkl(i,j,k,l) - 0.5e0*TT_ijkl(i,l,k,j) );

                	}
                }
            }
        }


        //DIIS attempt

        if(iter > 2) {

        	Error_Matrix = 	(F_p * P * S) - (S * P * F_p);
        	//cout << "\nError_Matrix:\n" << Error_Matrix << endl;
        }

        //Form Fock marix
        F = H + G;

        // Calculate the electronic energy
        electronic_energy = 0.5 * (P.cwiseProduct(H + F)).sum();

        // Transform the Fock matrix
        F_p = X.transpose() * F * X;

        // Diagonalize transformed Fock matrix
        //------------
        SelfAdjointEigenSolver<MatrixXd> es(F_p);

        E = es.eigenvalues().asDiagonal();
        C_p = es.eigenvectors();
        //------------

        // Return the original Coefficient matrix
        C = X * C_p;

        // Save present density matrix before creating new one
        P_old = P;

        // Calculate new density matrix: 2 * C * C^{T}
        // run over occupied orbitals only!
        P = 2*C.leftCols(nocc) * C.leftCols(nocc).transpose();

        // get deviation on the density matrix
        delta = sqrt( (P - P_old).array().square().sum() / 4);

        cout << "ITERATION NUMBER: " << iter + 1 << endl;
        cout << "\nF:\n" << F << endl;
        cout << "\nG:\n" << G << endl;
        cout << "\nF_p: \n" << F_p << endl;
        cout << "\nC_p: \n" << C_p << endl;
        cout << "\neigenvalues: \n" << E << endl;
        cout << "\nC: \n" << C << endl;
        cout << "\nP_old: \n" << P_old << endl;
        cout << "\nP: \n" << P << endl;
        cout << "\nenergy:\n" << electronic_energy << endl;
        cout << "\n DELTA(CONVERGENCE OF DENSITY MATRIX) =  " << delta <<  endl << endl;



        iter++;

	}

}
