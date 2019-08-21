#include "SOFIE_common.hxx"
#include "ROperator_Relu.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"
#include "ROperator.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace INTERNAL{
ROperator* make_ROperator_Relu(const onnx::NodeProto& nodeproto, RGraph& this_graph){
   ETensorType operator_type = this_graph.GetRDataNode(nodeproto.input(0))->GetType();
   switch(operator_type){
      case ETensorType::FLOAT:
         return new ROperator_Relu<float>(nodeproto, this_graph);
      default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + std::to_string(static_cast<int_t>(operator_type)));
   }
}
}//INTERNAL

template <typename T>
const std::vector<std::vector<int_t>> ROperator_Relu<T>::shapeInference() {
   //calculate output tensor shape
   std::vector<int_t> y_shape(X->GetShape());
   std::vector<std::vector<int_t>> ret;
   ret.push_back(std::move(y_shape));
   return ret;
}

template <typename T>
ROperator_Relu<T>::ROperator_Relu(const onnx::NodeProto& nodeproto, RGraph& this_graph):
X(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(0))))
{
   Y = new RDataNode<T>(shapeInference()[0], nodeproto.output(0));
   this_graph.RegisterNewRDataNode(Y);
}

template <typename T>
ROperator_Relu<T>::ROperator_Relu(const std::string& name_X, const std::string& name_Y, RGraph& this_graph):
X(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_X))),
Y(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_Y)))
{
   if (!X->HasSameShape(*Y)) throw std::runtime_error("TMVA::SOFIE Operator Relu has inputs of different shape");
}

template <typename T>
void ROperator_Relu<T>::Forward_reference(){
   OPERATION::Relu_reference(X->GetData(), Y->GetMutable(), Y->GetLength());
}

template <typename T>
void ROperator_Relu<T>::Forward_blas(){
   this->Forward_reference();
}


template class ROperator_Relu<float>;

}//SOFIE
}//Experimental
}//TMVA
