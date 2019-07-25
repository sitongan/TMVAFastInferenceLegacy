#ifndef TMVA_SOFIE_ROPERATOR_RELU
#define TMVA_SOFIE_ROPERATOR_RELU

#include "SOFIE_common.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace INTERNAL{
ROperator* make_ROperator_Relu(const onnx::NodeProto& nodeproto, RGraph& this_graph);
}//INTERNAL

namespace OPERATION{
//matrix multiplication A * B + C
//requires A has shape (m, k), B has shape (k, n), C has shape (m, n)
template <typename T>
void Relu_reference(const T* X, T* Y, int_t length)
{
   for (int i = 0; i < length; i++){
      Y[i] = ((X[i] > 0.0) ? X[i] : 0.0);
      //Y[i] = X[i];
   }
}
}//OPERATION

template <typename T>
class ROperator_Relu final : public ROperator
{

private:

   RDataNode<T>* X;
   RDataNode<T>* Y;


public:

   const std::vector<std::vector<int_t>> shapeInference() final;
   ROperator_Relu<T>(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   void Forward_reference() final;

};


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RELU
