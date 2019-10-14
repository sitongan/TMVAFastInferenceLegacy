#include "SOFIE_common.hxx"
#include "ROperator_Transpose.hxx"
#include "RDataNode.hxx"
#include "RGraph.hxx"
#include "ROperator.hxx"

#include "TMVA/RTensor.hxx"
#include "TMVA/DNN/Architectures/Cpu/CpuBuffer.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

namespace INTERNAL{
ROperator* make_ROperator_Transpose(const onnx::NodeProto& nodeproto, RGraph& this_graph){
   ETensorType operator_type = this_graph.GetRDataNode(nodeproto.input(0))->GetType();
   switch(operator_type){
      case ETensorType::FLOAT:
         return new ROperator_Transpose<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>(nodeproto, this_graph);
      default:
         throw std::runtime_error("TMVA::SOFIE Error - Unsupported - Operator Transpose does not yet support input type " + std::to_string(static_cast<int_t>(operator_type)));
   }
}
}//INTERNAL

template <typename T>
const std::vector<std::vector<size_t>> ROperator_Transpose<T>::shapeInference() {
   //calculate output tensor shape
   std::vector<size_t> transposed_shape;
   auto data_shape = data->GetShape();
   for (auto perm_value : attr_perm){
      transposed_shape.push_back(data_shape[perm_value]);
   }
   std::vector<std::vector<size_t>> ret;
   ret.push_back(std::move(transposed_shape));
   return ret;
}

template <typename T>
ROperator_Transpose<T>::ROperator_Transpose(const onnx::NodeProto& nodeproto, RGraph& this_graph):
data(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(nodeproto.input(0))))
{
   int input_dim = data->GetShape().size();
   if (nodeproto.attribute_size() == 1){
      attr_perm.assign(nodeproto.attribute(0).ints().begin(), nodeproto.attribute(0).ints().end());
      if (attr_perm.size() != input_dim) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - Transpose Operator has a perm attribute incompatible with input");
      for (auto perm_value : attr_perm){
         if (perm_value >= input_dim || perm_value < 0) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - Transpose Operator has a perm attribute incompatible with input");
      }
   }else{
      for (int i = input_dim - 1; i >= 0; i--){
         attr_perm.push_back(i);
      }
   }


   transposed = new RDataNode<T>(shapeInference()[0], nodeproto.output(0));
   this_graph.RegisterNewRDataNode(transposed);
}

template <typename T>
ROperator_Transpose<T>::ROperator_Transpose(const std::string& name_data, const std::string& name_transposed, const std::vector<int_t>& attribute_perm, RGraph& this_graph):
data(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_data))),
transposed(static_cast<RDataNode<T>*>(this_graph.GetRDataNode(name_transposed))),
attr_perm(attribute_perm)
{
   int input_dim = data->GetShape().size();
   if (attr_perm.size() != 0){
      if (attr_perm.size() != input_dim) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - Transpose Operator has a perm attribute incompatible with input");
      for (auto perm_value : attr_perm){
         if (perm_value >= input_dim || perm_value < 0) throw std::runtime_error("TMVA::SOFIE Error - Model Loading - Transpose Operator has a perm attribute incompatible with input");
      }
   }else{
      for (int i = input_dim - 1; i >= 0; i--){
         attr_perm.push_back(i);
      }
   }
}


template <typename T>
void ROperator_Transpose<T>::Forward_reference(){
   OPERATION::Transpose_reference(data->GetData(), data->GetShape(), transposed->GetData(), transposed->GetShape(), attr_perm);
}

template <typename T>
void ROperator_Transpose<T>::Forward_blas(){
   this->Forward_reference(); //tmp
}

template class ROperator_Transpose<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>;



}//SOFIE
}//Experimental
}//TMVA
