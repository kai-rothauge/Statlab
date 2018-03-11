#include "Statlab.hpp"

namespace statlab {

void delete1(El::Grid & grid, El::DistMatrix<Scalar> & A, El::DistMatrix<Scalar> & R, El::DistMatrix<Scalar> & Q, El::DistMatrix<Scalar> & H) {

	Int m = A.Height();			// Number of observations
	Int n = A.Width();			// Number of features

	El::DistMatrix<Scalar> RT(grid);
	El::DistMatrix<Scalar> AT(grid);
	El::Transpose(A, AT);
	El::Transpose(R, RT);

	El::LinearSolve(RT, AT);

	El::Transpose(AT, Q);

	Scalar s = 0.;
	Scalar q = 0.;

	for (Int i = 0; i < m; i++) {
		s = 0.;
		for (Int j = 0; j < n; j++) {
			q = Q.Get(i,j);
			s += q*q;
		}
		H.Set(i, 0, s);
	}
}

}
