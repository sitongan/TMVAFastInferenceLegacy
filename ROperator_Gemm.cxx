#include <string>
#include <unordered_map>
#include <utility>

#include "SOFIE_common.hxx"
#include "ROperator_Gemm.hxx"
#include "ROperator_Transpose.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"
#include "ROperator.hxx"

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
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + std::to_string(static_cast<int_t>(operator_type)));
   }
}
}//INTERNAL

template <typename T>
const std::vector<std::vector<int_t>> ROperator_Gemm<T>::shapeInference() {
   //calculate output tensor shape
   std::vector<int_t> y_shape;
   y_shape.push_back(attr_transA ? A->GetShape()[1] : A->GetShape()[0]);
   y_shape.push_back(attr_transB ? B->GetShape()[0] : B->GetShape()[1]);
   std::vector<std::vector<int_t>> ret;
   ret.push_back(std::move(y_shape));
   return ret;
}

template <typename T>
ROperator_Gemm<T>::ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph):
A(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(0)))),
B(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(1)))),
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
         if (attr_transA != 0 && attr_transA != 1) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - attribute transA in Operator Gemm not 0/1");
      }else if(attribute_name == "transB"){
         attr_transB = nodeproto.attribute(i).i();
         if (attr_transB != 0 && attr_transB != 1) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - attribute transB in Operator Gemm not 0/1");
      }else{
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attribute_name << " in OperatorNode " << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }
   if ((attr_transA ? A->GetShape()[0] : A->GetShape()[1]) != (attr_transB ? B->GetShape()[1] : B->GetShape()[0])){
       throw std::runtime_error("TMVA::SOFIE Error - Model Loading - input tensor A and B in Operator Gemm not compatible");
    }
   Y = new RDataNode<T>(shapeInference()[0], nodeproto.output(0));
   if ((Y->GetShape(0) != C->GetShape(0)) || (Y->GetShape(1) != C->GetShape(1))){
      C->Unidirectional_broadcast(Y->GetShape());  //brodcast and update C
   }
   this_graph.RegisterNewRDataNode(nodeproto.output(0), Y);
}

template <typename T>
ROperator_Gemm<T>::ROperator_Gemm(const std::string& name_A , const std::string& name_B, const std::string& name_C,\
   const std::string& name_Y, float attribute_alpha, float attribute_beta, int attribute_transA, int attribute_transB,\
   RGraph& this_graph):
A(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_A))),
B(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_B))),
C(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_C))),
Y(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_Y))),
attr_alpha(attribute_alpha),
attr_beta(attribute_beta),
attr_transA(attribute_transA),
attr_transB(attribute_transB)
{
   if ((attr_transA ? A->GetShape()[0] : A->GetShape()[1]) != (attr_transB ? B->GetShape()[1] : B->GetShape()[0])) {
       throw std::runtime_error("TMVA::SOFIE Error - Model Loading - input tensor A and B in Operator Gemm not compatible");
   }
   if ((attr_transA ? A->GetShape()[1] : A->GetShape()[0]) != Y->GetShape()[0]){
       throw std::runtime_error("TMVA::SOFIE Error - Model Loading - input tensor A and Y in Operator Gemm not compatible");
   }
   if ((attr_transB ? B->GetShape()[0] : B->GetShape()[1]) != Y->GetShape()[1]){
       throw std::runtime_error("TMVA::SOFIE Error - Model Loading - input tensor B and Y in Operator Gemm not compatible");
   }
   if ((Y->GetShape(0) != C->GetShape(0)) || (Y->GetShape(1) != C->GetShape(1))){
      C->Unidirectional_broadcast(Y->GetShape());  //brodcast and update C
   }
}




template <typename T>
void ROperator_Gemm<T>::Forward_reference(){
   if (attr_transA == 1){
      std::vector<T> transposed;
      transposed.resize(A->GetLength());
      auto old_shape = A->GetShape();
      OPERATION::Transpose_reference(A->GetData(), old_shape, transposed.data(), {old_shape[1], old_shape[0]}, {1,0});
      A->Update(std::move(transposed), {old_shape[1], old_shape[0]});
      attr_transA = 0;
   }
   if (attr_transB == 1){
      std::vector<T> transposed;
      transposed.resize(B->GetLength());
      auto old_shape = B->GetShape();
      OPERATION::Transpose_reference(B->GetData(), old_shape, transposed.data(), {old_shape[1], old_shape[0]}, {1,0});
      B->Update(std::move(transposed), {old_shape[1], old_shape[0]});
      attr_transB = 0;
   }
   OPERATION::Gemm_reference(A->GetData(), B->GetData(), C->GetData(), Y->GetMutable(), A->GetShape(0), A->GetShape(1), B->GetShape(1), attr_alpha, attr_beta);
}

template class ROperator_Gemm<float>;

}//SOFIE
}//Experimental
}//TMVA
