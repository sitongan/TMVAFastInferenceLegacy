#include <string>
#include <unordered_map>

#include "SOFIE.hxx"
#include "RDataNode.hxx"
#include "ROperator_Gemm.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


namespace INTERNAL{
ROperator* make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph){
   ETensorType operator_type = this_graph.GetRDataNode(nodeproto.input(0))->GetType();
   switch(operator_type){
      case ETensorType::FLOAT:
         return new ROperator_Gemm<float>(nodeproto, this_graph);
      default:
         throw std::runtime_error("Operator Gemm does not yet support input type " + std::to_string(static_cast<int_t>(operator_type)) + "\n");
   }
}
}//INTERNAL

template <typename T>
const std::vector<int_t> ROperator_Gemm<T>::outputShape() {
   //calculate output tensor shape and check tensor shape compatibility here
   std::vector<int_t> y_shape;
   y_shape.push_back(attr_transA ? A->GetShape()[1] : A->GetShape()[0]);
   y_shape.push_back(attr_transB ? B->GetShape()[0] : B->GetShape()[1]);
   return y_shape;
}

template <typename T>
ROperator_Gemm<T>::ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph):
A(static_cast<const RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(0)))),
B(static_cast<const RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(1)))),
C(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(2))))
{
   for (int i = 0; i < nodeproto.attribute_size(); i++){
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "alpha"){
         attr_alpha = nodeproto.attribute(i).f();
      }else if(attribute_name == "beta"){
         attr_beta = nodeproto.attribute(i).f();
      }else if(attribute_name == "transA"){
         attr_transA = nodeproto.attribute(i).i();
      }else if(attribute_name == "transB"){
         attr_transB = nodeproto.attribute(i).i();
      }else{
         std::cout << "TMVA::SOFIE Warning: Attribute " << attribute_name << " in OperatorNode " << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }
   Y = new RDataNode<T>(outputShape(), nodeproto.output(0));


   if ((Y->GetShape()[0] != C->GetShape()[0]) || (Y->GetShape()[1] != C->GetShape()[1])){
      UTILITY::unidirectional_broadcast(*C, Y->GetShape());
   }
   this_graph.RegisterNewRDataNode(nodeproto.output(0), Y);
}


template class ROperator_Gemm<float>;

}//SOFIE
}//Experimental
}//TMVA
