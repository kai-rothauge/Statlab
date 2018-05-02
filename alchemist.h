#ifndef ALCHEMIST__ALCHEMIST_H
#define ALCHEMIST__ALCHEMIST_H

#include <omp.h>
#include <El.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/mpi.hpp>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unistd.h>
#include <arpa/inet.h>
#include <eigen3/Eigen/Dense>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
// #include "spdlog/fmt/ostr.h"
#include "endian.h"

#ifdef ALDEBUG
#define ENSURE(x) assert(x)
#define ALCHEMIST_TRACE(x) do { x; } while(0)
#else
#define ENSURE(x) x
#define ALCHEMIST_TRACE(x) do { } while(0)
#endif

namespace alchemist {

namespace serialization = boost::serialization;
namespace mpi = boost::mpi;
using boost::format;

typedef El::Matrix<double> Matrix;
typedef El::AbstractDistMatrix<double> DistMatrix;
typedef uint32_t WorkerId;


void kmeansPP(uint32_t seed, std::vector<Eigen::MatrixXd> points, std::vector<double> weights,
    Eigen::MatrixXd & fitCenters, uint32_t maxIters);

struct Worker;

struct MatrixHandle {
  uint32_t id;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & id;
  }
};

inline bool operator < (const MatrixHandle &lhs, const MatrixHandle &rhs) {
  return lhs.id < rhs.id;
}

struct MatrixDescriptor {
  MatrixHandle handle;
  size_t numRows;
  size_t numCols;

  explicit MatrixDescriptor() :
      numRows(0), numCols(0) {
  }

  MatrixDescriptor(MatrixHandle handle, size_t numRows, size_t numCols) :
      handle(handle), numRows(numRows), numCols(numCols) {
  }

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & handle;
    ar & numRows;
    ar & numCols;
  }
};

struct WorkerInfo {
  std::string hostname;
  uint32_t port;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & hostname;
    ar & port;
  }
};

struct Command {
  virtual ~Command() {
  }

  virtual void run(Worker *self) const = 0;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
  }
};

struct HaltCommand : Command {
  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
  }
};


/*
struct ThinSVDCommand : Command {
  MatrixHandle mat;
  uint32_t whichFactors;
  uint32_t krank;
  MatrixHandle U;
  MatrixHandle S;
  MatrixHandle V;
  explicit ThinSVDCommand() {}
  ThinSVDCommand(MatrixHandle mat, uint32_t whichFactors, uint32_t krank,
      MatrixHandle U, MatrixHandle S, MatrixHandle V) :
    mat(mat), whichFactors(whichFactors), krank(krank), U(U), S(S), V(V) {}
  virtual void run(Worker *self) const;
  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_oject<Command>(*this);
    ar & mat;
    ar & whichFactors;
    ar & krank;
    ar & U;
    ar & S;
    ar & V;
  }
}
*/

struct TransposeCommand : Command {
  MatrixHandle origMat;
  MatrixHandle transposeMat;

  explicit TransposeCommand() {}

  TransposeCommand(MatrixHandle origMat, MatrixHandle transposeMat) :
    origMat(origMat), transposeMat(transposeMat) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & origMat;
    ar & transposeMat;
  }
};

struct SkylarkKernelSolverCommand : Command {
    MatrixHandle features;
    MatrixHandle targets;
    MatrixHandle coefs;
    bool regression; // regression by default (false for classification)
    uint32_t lossfunction;
    uint32_t regularizer;
    uint32_t kernel;
    double kernelparam;
    double kernelparam2;
    double kernelparam3;
    double lambda;
    uint32_t maxiter;
    double tolerance;
    double rho;
    uint32_t seed;
    uint32_t randomfeatures;
    uint32_t numfeaturepartitions;

    explicit SkylarkKernelSolverCommand() {}

    SkylarkKernelSolverCommand(MatrixHandle features, MatrixHandle targets, MatrixHandle coefs, bool regression,
        uint32_t lossfunction, uint32_t regularizer, uint32_t kernel,
        double kernelparam, double kernelparam2, double kernelparam3,
        double lambda, uint32_t maxiter, double tolerance, double rho,
        uint32_t seed, uint32_t randomfeatures, uint32_t numfeaturepartitions) :
        features(features), targets(targets), coefs(coefs), regression(regression),
        lossfunction(lossfunction), regularizer(regularizer), kernel(kernel),
        kernelparam(kernelparam), kernelparam2(kernelparam2), kernelparam3(kernelparam3),
        lambda(lambda), maxiter(maxiter), tolerance(tolerance), rho(rho),
        seed(seed), randomfeatures(randomfeatures), numfeaturepartitions(numfeaturepartitions) {
        }

    virtual void run(Worker *self) const;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned version) {
        ar & serialization::base_object<Command>(*this);
        ar & features;
        ar & targets;
        ar & coefs;
        ar & regression;
        ar & lossfunction;
        ar & regularizer;
        ar & kernel;
        ar & kernelparam;
        ar & kernelparam2;
        ar & kernelparam3;
        ar & lambda;
        ar & maxiter;
        ar & tolerance;
        ar & rho;
        ar & seed;
        ar & randomfeatures;
        ar & numfeaturepartitions;
    }
};

struct SkylarkLSQRSolverCommand : Command {
  MatrixHandle A;
  MatrixHandle B;
  MatrixHandle X;
  double tolerance;
  uint32_t iter_lim;

  explicit SkylarkLSQRSolverCommand() {}

  SkylarkLSQRSolverCommand(MatrixHandle A, MatrixHandle B, MatrixHandle X,
      double tolerance, uint32_t iter_lim):
    A(A), B(B), X(X), tolerance(tolerance), iter_lim(iter_lim) {};

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & A;
    ar & B;
    ar & X;
    ar & tolerance;
    ar & iter_lim;
  }
};

struct RandomFourierFeaturesCommand : Command {
    MatrixHandle A;
    MatrixHandle X;
    uint32_t numRandFeatures;
    double sigma;
    uint32_t seed;

    explicit RandomFourierFeaturesCommand() {}

    RandomFourierFeaturesCommand(MatrixHandle A, MatrixHandle X, uint32_t numRandFeatures, double sigma, uint32_t seed):
        A(A), X(X), numRandFeatures(numRandFeatures), sigma(sigma), seed(seed) {};

    virtual void run(Worker * self) const;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned version) {
        ar & serialization::base_object<Command>(*this);
        ar & A;
        ar & X;
        ar & numRandFeatures;
        ar & sigma;
        ar & seed;
    }
};

struct ReadHDF5Command : Command {
    MatrixHandle A;
    std::string fname;
    std::string varname;
    int colreplicas;

    explicit ReadHDF5Command() {}

    ReadHDF5Command(MatrixHandle A, std::string fname, std::string varname, int colreplicas):
        A(A), fname(fname), varname(varname), colreplicas(colreplicas) {};

    virtual void run(Worker * self) const;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned version) {
        ar & serialization::base_object<Command>(*this);
        ar & A;
        ar & fname;
        ar & varname;
        ar & colreplicas;
    }

};

struct NormalizeMatInPlaceCommand : Command {
    MatrixHandle A;

    explicit NormalizeMatInPlaceCommand() {}

    NormalizeMatInPlaceCommand(MatrixHandle A): A(A) {};

    virtual void run(Worker *self) const;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned version) {
        ar & serialization::base_object<Command>(*this);
        ar & A;
    }

};

struct FactorizedCGSolverCommand : Command {
  MatrixHandle A;
  MatrixHandle B;
  MatrixHandle X;
  double lambda;
  uint32_t maxIters;

  explicit FactorizedCGSolverCommand() {}

  FactorizedCGSolverCommand(MatrixHandle A, MatrixHandle B, MatrixHandle X, double lambda, uint32_t maxIters):
    A(A), B(B), X(X), lambda(lambda), maxIters(maxIters) {};

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & A;
    ar & B;
    ar & X;
    ar & lambda;
    ar & maxIters;
  }
};

struct KMeansCommand : Command {
  MatrixHandle origMat;
  uint32_t numCenters;
  uint32_t initSteps; // relevant in k-means|| only
  double changeThreshold; // stop when all centers change by Euclidean distance less than changeThreshold
  uint32_t method;
  uint64_t seed;
  MatrixHandle centersHandle;
  MatrixHandle assignmentsHandle;

  explicit KMeansCommand() {}

  KMeansCommand(MatrixHandle origMat, uint32_t numCenters, uint32_t method,
      uint32_t initSteps, double changeThreshold, uint64_t seed,
      MatrixHandle centersHandle, MatrixHandle assignmentsHandle) :
    origMat(origMat), numCenters(numCenters), method(method),
    initSteps(initSteps), changeThreshold(changeThreshold),
    seed(seed), centersHandle(centersHandle), assignmentsHandle(assignmentsHandle) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & origMat;
    ar & numCenters;
    ar & initSteps;
    ar & changeThreshold;
    ar & method;
    ar & seed,
    ar & centersHandle;
    ar & assignmentsHandle;
  }
};

struct TruncatedSVDCommand : Command {
  MatrixHandle mat;
  MatrixHandle UHandle;
  MatrixHandle SHandle;
  MatrixHandle VHandle;
  uint32_t k;
  uint32_t method;

  explicit TruncatedSVDCommand() {}

  TruncatedSVDCommand(MatrixHandle mat, MatrixHandle UHandle,
      MatrixHandle SHandle, MatrixHandle VHandle, uint32_t k, uint32_t method) :
    mat(mat), UHandle(UHandle), SHandle(SHandle), VHandle(VHandle),
    k(k), method(method) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & UHandle;
    ar & SHandle;
    ar & VHandle;
    ar & k;
    ar & method;
  }
};

struct LeverageScoreCommand : Command {
  MatrixHandle mat;
  MatrixHandle SHandle;
  uint32_t k;
  uint32_t method;

  explicit LeverageScoreCommand() {}

  LeverageScoreCommand(MatrixHandle _mat, MatrixHandle _SHandle,
		  uint32_t k, uint32_t method) : mat(_mat), SHandle(_SHandle), k(k), method(method) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & SHandle;
    ar & k;
    ar & method;
  }
};

struct LeastAbsoluteDeviationsCommand : Command {
  MatrixHandle Amat;
  MatrixHandle bvec;
  MatrixHandle xvec;

  explicit LeastAbsoluteDeviationsCommand() {}

  LeastAbsoluteDeviationsCommand(MatrixHandle Amat, MatrixHandle bvec, MatrixHandle xvec):
    Amat(Amat), bvec(bvec), xvec(xvec) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & Amat;
    ar & bvec;
    ar & xvec;
  }
};

struct ThinSVDCommand : Command {
  MatrixHandle mat;
  MatrixHandle Uhandle;
  MatrixHandle Shandle;
  MatrixHandle Vhandle;

  explicit ThinSVDCommand() {}

  ThinSVDCommand(MatrixHandle mat, MatrixHandle Uhandle,
      MatrixHandle Shandle, MatrixHandle Vhandle) :
    mat(mat), Uhandle(Uhandle), Shandle(Shandle), Vhandle(Vhandle) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & Uhandle;
    ar & Shandle;
    ar & Vhandle;
  }
};

struct MatrixMulCommand : Command {
  MatrixHandle handle;
  MatrixHandle inputA;
  MatrixHandle inputB;

  explicit MatrixMulCommand() {}

  MatrixMulCommand(MatrixHandle dest, MatrixHandle A, MatrixHandle B) :
    handle(dest), inputA(A), inputB(B) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
    ar & inputA;
    ar & inputB;
  }
};

struct MatrixGetRowsCommand : Command {
  MatrixHandle handle;

  explicit MatrixGetRowsCommand() {}

  MatrixGetRowsCommand(MatrixHandle handle) :
    handle(handle) {}

  virtual void run(Worker * self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
  }
};

struct MatrixGetWorkerRowsCommand : Command {
  MatrixHandle handle;

  explicit MatrixGetWorkerRowsCommand() {}

  MatrixGetWorkerRowsCommand(MatrixHandle handle) :
    handle(handle) {}

  virtual void run(Worker * self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
  }
};

struct NewMatrixCommand : Command {
  MatrixDescriptor info;

  explicit NewMatrixCommand() {
  }

  NewMatrixCommand(const MatrixDescriptor &info) :
    info(info) {
  }

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & info;
  }
};

int driverMain(const mpi::communicator &world, int argc, char *argv[]);
int workerMain(const mpi::communicator &world, const mpi::communicator &peers);

} // namespace alchemist

namespace fmt {
  // for displaying Eigen expressions. Note, if you include spdlog/fmt/ostr.h, this will be
  // hidden by the ostream<< function for Eigen objects
  template <typename Formatter, typename Derived>
  inline void format_arg(Formatter &f,
      const char *&format_str, const Eigen::MatrixBase<Derived> &exp) {
    std::stringstream buf;
    buf << "Eigen matrix " << std::endl << exp;
    f.writer().write("{}", buf.str());
  }

  template <typename Formatter>
  inline void format_arg(Formatter &f,
      const char *&format_str, const Eigen::Matrix<double, -1, -1> &exp) {
    std::stringstream buf;
    buf << "Eigen matrix " << std::endl << exp;
    f.writer().write("{}", buf.str());
  }

  // for displaying vectors
  template <typename T, typename A>
  inline void format_arg(BasicFormatter<char> &f,
      const char *&format_str, const std::vector<T,A> &vec) {
    std::stringstream buf;
    buf << "Vector of length " << vec.size() << std::endl << "{";
    for(typename std::vector<T>::size_type pos=0; pos < vec.size()-1; ++pos) {
      buf << vec[pos] << "," << std::endl;
    }
    buf << vec[vec.size()-1] << "}";
    f.writer().write("{}", buf.str());
  }

  inline void format_arg(BasicFormatter<char> &f,
      const char *&format_str, const alchemist::MatrixHandle &handle) {
    f.writer().write("[{}]", handle.id);
  }
}

namespace boost { namespace serialization {
  // to serialize Eigen Matrix objects
	template< class Archive,
						class S,
						int Rows_,
						int Cols_,
						int Ops_,
						int MaxRows_,
						int MaxCols_>
	inline void serialize(Archive & ar,
		Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix,
		const unsigned int version)
	{
		int rows = matrix.rows();
		int cols = matrix.cols();
		ar & make_nvp("rows", rows);
		ar & make_nvp("cols", cols);
		matrix.resize(rows, cols); // no-op if size does not change!

		// always save/load col-major
		for(int c = 0; c < cols; ++c)
			for(int r = 0; r < rows; ++r)
				ar & make_nvp("val", matrix(r,c));
	}
}} // namespace boost::serialization

namespace alchemist {

struct Worker {
  WorkerId id;
  mpi::communicator world;
  mpi::communicator peers;
  El::Grid grid;
  bool shouldExit;
  int listenSock;
  std::map<MatrixHandle, std::unique_ptr<DistMatrix>> matrices;
  std::shared_ptr<spdlog::logger> log;

  Worker(const mpi::communicator &world, const mpi::communicator &peers) :
      id(world.rank() - 1), world(world), peers(peers), grid(El::mpi::Comm(peers)),
      shouldExit(false), listenSock(-1) {
    ENSURE(peers.rank() == world.rank() - 1);
  }

  void receiveMatrixBlocks(MatrixHandle handle);
  void sendMatrixRows(MatrixHandle handle, const El::AbstractDistMatrix<double> * matrix);
  int main();
};

}

BOOST_CLASS_EXPORT_KEY(alchemist::MatrixDescriptor);
BOOST_CLASS_EXPORT_KEY(alchemist::Command);
BOOST_CLASS_EXPORT_KEY(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixMulCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixGetWorkerRowsCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixGetRowsCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::ThinSVDCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::TransposeCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::KMeansCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::TruncatedSVDCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::LeverageScoreCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::LeastAbsoluteDeviationsCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::SkylarkKernelSolverCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::SkylarkLSQRSolverCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::FactorizedCGSolverCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::RandomFourierFeaturesCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::ReadHDF5Command);
BOOST_CLASS_EXPORT_KEY(alchemist::NormalizeMatInPlaceCommand);

#endif
