#ifndef TMVA_SOFIE_ROPERATOR_TRANSPOSE
#define TMVA_SOFIE_ROPERATOR_TRANSPOSE

#include "SOFIE_common.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"

#include <vector>

namespace TMVA{
namespace Experimental{
namespace SOFIE{


namespace INTERNAL{
ROperator* make_ROperator_Transpose(const onnx::NodeProto& nodeproto, RGraph& this_graph);
}//INTERNAL

namespace OPERATION{
//matrix transpose
template <typename T>
void Transpose_reference(const T* input, const std::vector<int_t>& shape_input, T* transposed, const std::vector<int_t>& shape_transposed, const std::vector<int_t>& perm)
{
   int_t length = 1;
   for (auto dim: shape_input) length *= dim;
   std::vector<int_t> old_indices(shape_input.size(), 0);
   for (int_t position = 0; position < length; position++){
      std::vector<int_t> new_indices(shape_input.size(), 0);
      std::vector<int_t> old_indices = UTILITY::Position_to_indices(position,shape_input);
      for (int dim = 0; dim < shape_input.size(); dim++){
         new_indices[dim] = old_indices[perm[dim]];
      }
      transposed[UTILITY::Indices_to_position(new_indices, shape_transposed)] = input[position];
   }
}
}//OPERATION



template <typename T>
class ROperator_Transpose final : public ROperator
{

private:

   std::vector<int_t> attr_perm;

   RDataNode<T>* data;
   RDataNode<T>* transposed;

public:

   const std::vector<std::vector<int_t>> shapeInference() final;
   ROperator_Transpose<T>(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   void Forward_reference() final;

};





}//SOFIE
}//Experimental
}//TMVA







#endif //TMVA_SOFIE_ROPERATOR_GEMM
