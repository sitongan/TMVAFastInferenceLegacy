#ifndef TMVA_SOFIE_ROPERATOR_GEMM
#define TMVA_SOFIE_ROPERATOR_GEMM

#include "SOFIE_common.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


namespace INTERNAL{
ROperator* make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
}//INTERNAL

namespace OPERATION{
//matrix multiplication A * B + C
//requires A has shape (m, k), B has shape (k, n), C has shape (m, n)
template <typename T>
void Gemm_reference(const T* A, const T* B, const T* C, T* Y, int_t m, int_t k, int_t n, float alpha = 1.0, float beta = 1.0)
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

   const std::vector<std::vector<int_t>> shapeInference() final;
   ROperator_Gemm<T>(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   void Forward_reference() final;

};





}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_GEMM