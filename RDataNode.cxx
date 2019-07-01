
#include "RDataNode.hxx"
#include <cstring>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
RDataNode<T>::RDataNode(const onnx::TensorProto& tensorproto)
{
   set_fType();
   if (tensorproto.has_name()){
      fName = tensorproto.name();
   }
   fLength = 1;
   for (int i = 0; i < tensorproto.dims_size(); i++){
      fShape.push_back(tensorproto.dims(i));
      fLength *= tensorproto.dims(i);
   }

   if (tensorproto.has_data_location() && tensorproto.data_location() == onnx::TensorProto::EXTERNAL ){
      throw std::runtime_error("Tensors with externally stored weights have not been supported yet.");
   }

   if (tensorproto.has_raw_data()){
      //const float* raw_data = reinterpret_cast<const float*>(tensorproto.raw_data().c_str());
      //fDataVector = new std::vector<float>(raw_data, raw_data+fLength);
      //fHasData = true;
      fImmutableData = reinterpret_cast<const T*>(tensorproto.raw_data().c_str());
      fHasImmutableData = true;
   }else{
      get_protobuf_datafield(tensorproto);
      fHasImmutableData = true;
   }

   fIsSegment = tensorproto.has_segment();
   if (fIsSegment){
      fSegmentIndex = std::make_tuple(tensorproto.segment().begin(),tensorproto.segment().end());
   }
}

template <typename T>
RDataNode<T>::RDataNode(const std::vector<int_t>& shape, const std::string& name) :fName(name){
   set_fType();
   fShape.assign(shape.begin(), shape.end());
   fLength = 1;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   fDataVector = new std::vector<T>();
   fDataVector->reserve(fLength);
   fHasData = true;
}

template <typename T>
RDataNode<T>::RDataNode(const onnx::ValueInfoProto& valueinfoproto, const std::unordered_map<std::string, int_t>& dimensionDenotationMap){
   set_fType();
   if (valueinfoproto.has_name()){
      fName = valueinfoproto.name();
   }else{
      fName = "";
   }
   fLength = 1;
   for (int i = 0; i < valueinfoproto.type().tensor_type().shape().dim_size(); i++){
      int_t dim;
      if (valueinfoproto.type().tensor_type().shape().dim(i).has_dim_value()){
         dim = valueinfoproto.type().tensor_type().shape().dim(i).dim_value();
      }else if (valueinfoproto.type().tensor_type().shape().dim(i).has_dim_param()){
         auto dimensionDenotation = dimensionDenotationMap.find(valueinfoproto.type().tensor_type().shape().dim(i).dim_param());
         if (dimensionDenotation != dimensionDenotationMap.end()){
            dim = dimensionDenotation->second;
         }else{
            throw std::runtime_error("Dimension Denotation " + dimensionDenotation->first + " not found in graph specification!\n");
         }
      }else{
         throw std::runtime_error("ONNX file error: Valueinfoproto " + fName + " has neither dim_value nor dim_param! \n");
      }
      fShape.push_back(dim);
      fLength *= dim;
   }
   fDataVector = new std::vector<T>();
   fDataVector->reserve(fLength);
   fHasData = true;
}

template <typename T>
RDataNode<T>::RDataNode(const std::vector<T>& input, const std::vector<int_t>& shape, const std::string& name) :fName(name){
   set_fType();
   fShape.assign(shape.begin(), shape.end());
   fLength = input.size();
   fDataVector = new std::vector<T>(input);
   fHasData = true;
}

template <typename T>
RDataNode<T>::~RDataNode(){
   if (fHasData){
      delete fDataVector;
   }
}

template <typename T>
const T* RDataNode<T>::GetData(){
   if (fHasImmutableData){
      return fImmutableData;
   }else if (fHasData){
      return fDataVector->data();
   }else{
      throw std::runtime_error("Tensor " + fName + " has no data.");
   }
}

template <typename T>
void RDataNode<T>::Update(std::vector<T>& newDataVector, std::vector<int_t> newShape){
   if (fHasImmutableData){
      fHasImmutableData = false;
   }else if(fHasData){
      delete fDataVector;
   }
   fDataVector = new std::vector<T>(std::move(newDataVector));
   fHasData = true;
   fShape.assign(newShape.begin(), newShape.end());
   fLength = 1;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
}

template class RDataNode<float>;   //explicit template initialization






}//SOFIE
}//Experimental
}//TMVA
