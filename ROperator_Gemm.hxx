#ifndef TMVA_SOFIE_ROPERATOR_GEMM
#define TMVA_SOFIE_ROPERATOR_GEMM


#include "SOFIE.hxx"
#include "ROperator.hxx"
#include "RDataNode.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

//necessary forward declaration
//class ROperator;
class RGraph;

/*
namespace INTERNAL{
ROperator* make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
}//INTERNAL*/


template <typename T>
class ROperator_Gemm final : public ROperator
{

private:

   float attr_alpha = 1.0;
   float attr_beta = 1.0;
   int_t attr_transA = 0;
   int_t attr_transB = 0;

   const RDataNode<T>* A;
   const RDataNode<T>* B;
   RDataNode<T>* C;  //might need to be broadcast
   RDataNode<T>* Y;

   template<class Q = T>
   typename std::enable_if<std::is_same<Q, float>::value, void>::type forward_fallback_typed()
   {
      //todo
   }

public:

   const std::vector<int_t> outputShape() final;
   ROperator_Gemm<T>(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   void forward_fallback() final { forward_fallback_typed(); }

};





}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_GEMM
