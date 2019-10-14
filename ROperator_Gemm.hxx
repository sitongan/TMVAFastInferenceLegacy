#ifndef TMVA_SOFIE_ROPERATOR_GEMM
#define TMVA_SOFIE_ROPERATOR_GEMM

#include "SOFIE_common.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"
#include "ROperator.hxx"

#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace BLAS{

extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
                       const float * beta, float * C, const int * ldc);
}//BLAS


namespace INTERNAL{
ROperator* make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
}//INTERNAL

namespace OPERATION{
//matrix multiplication A * B + C
//requires A has shape (m, k), B has shape (k, n), C has shape (m, n)
template <typename T>
void Gemm_reference(const T* A, const T* B, const T* C, T* Y, int_t m, int_t n, int_t k, float alpha = 1.0, float beta = 1.0)
{

   T entry_sum;
   for(int mm=0; mm < m; mm++){
      for (int nn=0; nn < n; nn++){
         entry_sum = 0;
         for (int kk=0; kk < k; kk++){
            entry_sum += alpha * A[mm * k + kk] * B[kk * n + nn];
         }
         Y[mm * n + nn] = entry_sum + beta * C[mm * n + nn];
      }
   }
}
}//OPERATION



template <typename T>
class ROperator_Gemm final : public ROperator
{

private:

   float attr_alpha = 1.0;
   float attr_beta = 1.0;
   int_t attr_transA = 0;
   int_t attr_transB = 0;

   RDataNode<T>* A;
   RDataNode<T>* B;
   RDataNode<T>* C;
   RDataNode<T>* Y;

public:

   const std::vector<std::vector<size_t>> shapeInference() final;
   ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   ROperator_Gemm(const std::string& name_A , const std::string& name_B, const std::string& name_C,
      const std::string& name_Y, float attribute_alpha, float attribute_beta, int attribute_transA, int attribute_transB,
      RGraph& this_graph);
   void Forward_reference() final;
   void Forward_blas() final;

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_GEMM
