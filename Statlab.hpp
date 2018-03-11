#include <string>
#include <chrono>
#include <El.hpp>

using El::Int;

namespace statlab {

	typedef double Real;
	typedef double Scalar;


	void leverage(const El::Grid & grid, El::DistMatrix<Scalar> & A, El::DistMatrix<Scalar> & R, El::DistMatrix<Scalar> & Q, El::DistMatrix<Scalar> & H);

	void delete1(El::Grid & grid, El::DistMatrix<Scalar> & A, El::DistMatrix<Scalar> & R, El::DistMatrix<Scalar> & Q, El::DistMatrix<Scalar> & H);
}
