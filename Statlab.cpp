#include "Statlab.hpp"

using namespace statlab;

int main(int argc, char *argv[]) {

	El::Environment env( argc, argv );
	El::mpi::Comm comm = El::mpi::COMM_WORLD;

	try
	{
		El::Int m = El::Input("--height","height of matrix",100);
		El::Int n = El::Input("--width","width of matrix",100);
		const El::Int nb = El::Input("--nb","blocksize",96);
		const bool print = El::Input("--print","print matrices?",false);
		El::ProcessInput();
		El::PrintInputReport();

//		const Int m = 10;
//		const Int n = 6;
//		const Int nb = 5;
//		const bool print = true;

		El::SetBlocksize( nb );

		auto t1 = std::chrono::high_resolution_clock::now();
		const El::Grid grid( comm );
		El::DistMatrix<Scalar> A(grid);
		El::Uniform( A, m, n );
		auto t2 = std::chrono::high_resolution_clock::now();

		El::Output("Initialization time: ", std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());

		t1 = std::chrono::high_resolution_clock::now();
		for (Int i = 0; i < m; i++)
			A.Set(i,0,1.);
		t2 = std::chrono::high_resolution_clock::now();
		El::Output("Set time: ", std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());

//		El::Read(A, "A1.txt", El::AUTO, false);
//		m = A.Height();
//		n = A.Width();

		if( print )
			El::Print( A, "A" );
		const Real frobA = El::FrobeniusNorm( A );

		// Compute the QR decomposition of A, but do not overwrite A
		El::DistMatrix<Scalar> Q( A ), QFull( A ), R;
		R.SetGrid(grid);
		t1 = std::chrono::high_resolution_clock::now();
		El::qr::Explicit( Q, R );
		t2 = std::chrono::high_resolution_clock::now();
//		El::qr::Explicit( QFull, R, false );
		if( print )
		{
			El::Print( Q, "Q" );
//			El::Print( QFull, "QFull" );
			El::Print( R, "R" );
		}
		El::Output("QR time: ", std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());

		// Check the error in the QR factorization, || A - Q R ||_F / || A ||_F
		t1 = std::chrono::high_resolution_clock::now();
		El::DistMatrix<Scalar> E( A );
		El::Output(E.Height()," ",E.Width()," ",Q.Height()," ",Q.Width()," ",R.Height()," ",R.Width());
		El::Gemm( El::NORMAL, El::NORMAL, Scalar(-1), Q, R, Scalar(1), E );
		const Real frobQR = El::FrobeniusNorm( E );
		t2 = std::chrono::high_resolution_clock::now();
		El::Output("Check 1 time: ", std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());

		// Check the numerical orthogonality of Q, || I - Q^H Q ||_F / || A ||_F
		t1 = std::chrono::high_resolution_clock::now();
		const El::Int k = El::Min(m,n);
		El::Identity( E, k, k );
		El::Herk( El::LOWER, El::ADJOINT, Real(-1), Q, Real(1), E );
		const Real frobOrthog = El::HermitianFrobeniusNorm( El::LOWER, E );
		t2 = std::chrono::high_resolution_clock::now();
		El::Output("Check 2 time: ", std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());

		if( El::mpi::Rank(comm) == 0 )
			El::Output
			("|| A ||_F = ",frobA,"\n",
			 "|| A - Q R ||_F / || A ||_F   = ",frobQR/frobA,"\n",
			 "|| I - Q^H Q ||_F / || A ||_F = ",frobOrthog/frobA,"\n");


		El::DistMatrix<Scalar> H(grid);
		El::Zeros(H, m, 1);

		t1 = std::chrono::high_resolution_clock::now();
		leverage(grid, A, R, Q, H);
		t2 = std::chrono::high_resolution_clock::now();
		El::Output("Leverage time: ", std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());

//		El::Print(H);

	}
	catch( std::exception& e ) { El::ReportException(e); }

	return 0;
}
