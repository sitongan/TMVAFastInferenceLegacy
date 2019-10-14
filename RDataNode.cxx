
#include "RDataNode.hxx"
#include "TMVA/RTensor.hxx"
#include "TMVA/DNN/Architectures/Cpu/CpuBuffer.h"

#include <cstring>
#include <cstdlib>
#include <memory>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

//constructor for immutable tensorproto (weights)
template <typename T>
RDataNode<T>::RDataNode(onnx::TensorProto* tensorproto)
{
   set_fType();
   if (tensorproto->has_name()){
      fName = tensorproto->name();
   }
   fLength = 1;
   for (int i = 0; i < tensorproto->dims_size(); i++){
      fShape.push_back(tensorproto->dims(i));
      fLength *= tensorproto->dims(i);
   }

   if (tensorproto->has_data_location() && tensorproto->data_location() == onnx::TensorProto::EXTERNAL ){
      throw std::runtime_error("Tensors with externally stored weights have not been supported yet.");
   }

   fDataTensor = new T(fShape);

   if (tensorproto->has_raw_data()){

      auto raw_data_ptr = reinterpret_cast<Value_t*>(const_cast<char*>(tensorproto->mutable_raw_data()->c_str()));   //tensorproto retains ownership
      std::memcpy(fDataTensor->GetData(), raw_data_ptr, fLength * sizeof(Value_t));
      //fDataVector = new std::vector<T>(raw_data_ptr, raw_data_ptr + fLength); //copy here
   }else{

      extract_protobuf_datafield(tensorproto);
   }

   fIsSegment = tensorproto->has_segment();
   if (fIsSegment){
      fSegmentIndex = std::make_tuple(tensorproto->segment().begin(),tensorproto->segment().end());
      throw std::runtime_error("Tensors with segments have not been supported yet.");
   }
}


//constructor for mutable tensors (shape from valueinfoproto)
template <typename T>
RDataNode<T>::RDataNode(const onnx::ValueInfoProto& valueinfoproto, const std::unordered_map<std::string, int_t>& dimensionDenotationMap){
   set_fType();
   if (valueinfoproto.has_name()){
      fName = valueinfoproto.name();
   }else{
      fName = "";
   }
   fLength = 1;
   if (!valueinfoproto.type().tensor_type().has_shape()) throw std::runtime_error("TMVA::SOFIE datanode with no shape restrictions is not supported yet");
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
   if (valueinfoproto.type().tensor_type().shape().dim_size() == 0) fShape = {1};   //in case this TensorShapeProto has no dimension message: ONNX IR defines this to be a scalar

   //fDataVector = new std::vector<T>();
   //fDataVector->resize(fLength);
   fDataTensor = new T(fShape);
}

//construct from a std vector, copy data
template <typename T>
RDataNode<T>::RDataNode(const std::vector<T>& input, const std::vector<size_t>& shape, const std::string& name) :RDataNodeBase(name){
   set_fType();
   fShape = shape;
   fLength = input.size();
   //fDataVector = new std::vector<T>(input);
   fDataTensor = new T(fShape);
   std::memcpy(fDataTensor->GetData(), input.data(), fLength * sizeof(Value_t));
}

//construct from a r-value std vector, move data
template <typename T>
RDataNode<T>::RDataNode(Container_t&& input, int_t size, const std::vector<size_t>& shape, const std::string& name) :RDataNodeBase(name){
   set_fType();
   fShape = shape;
   fLength = size;
   //fDataVector = new std::vector<T>(input);
   fDataTensor = new T(std::make_shared<Container_t>(input), fShape);
}

//constructor by shape and type
template <typename T>
RDataNode<T>::RDataNode(const std::vector<std::size_t>& shape, const std::string& name) :RDataNodeBase(name){
   set_fType();
   fShape = shape;
   fLength = 1;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   //fDataVector = new std::vector<T>();
   //fDataVector->resize(fLength);
   fDataTensor = new T(fShape);
}




template <typename T>
RDataNode<T>::~RDataNode(){
   //delete fDataVector;
   delete fDataTensor;
}

template <typename T>
const typename T::Value_t* RDataNode<T>::GetData() const{
   //return fDataVector->data();
   return fDataTensor->GetData();
}

template <typename T>
typename T::Value_t* RDataNode<T>::GetData(){
   return fDataTensor->GetData();
}


template <typename T>
void RDataNode<T>::Update(Container_t&& newDataVector, int_t newSize, const std::vector<size_t>& newShape){
   fLength = 1;
   fShape = newShape;   //copy assignment
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   if (fLength != newSize) throw std::runtime_error("TMVA::SOFIE - RDataNode Update Error, size inconsistency");

   //*fDataVector = newDataVector;  //this will be move assignment
   delete fDataTensor;
   fDataTensor = new T(std::make_shared<Container_t>(newDataVector), fShape);
}

template <typename T>
void RDataNode<T>::Update(const std::vector<Value_t>& newDataVector,const std::vector<size_t>& newShape){
   fLength = 1;
   fShape = newShape;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   if (fLength != newDataVector.size()) throw std::runtime_error("TMVA::SOFIE - RDataNode Update Error, size inconsistency");

   //*fDataVector = newDataVector;  //this will be copy assignment
   delete fDataTensor;
   fDataTensor = new T (fShape);
   std::memcpy(fDataTensor->GetData(), newDataVector.data(), fLength * sizeof(Value_t));
}

template <typename T>
RDataNode<T>& RDataNode<T>::operator=(const RDataNode<T>& other){
   //delete fDataVector;
   delete fDataTensor;
   fShape = other.GetShape(); //copy
   fDataTensor = new T(fShape);
   std::memcpy(fDataTensor->GetData(), other.GetData(), other.GetLength() * sizeof(Value_t));
   //fDataVector = new std::vector<T>(other.GetData(), other.GetData() + other.GetLength());   //copy
   fLength = 1;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   return *this;
}


//template class RDataNode<float>;   //explicit template initialization
template class RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>;   //explicit template initialization




namespace{
template<typename T>
static inline void copy_vector_data(int_t no_of_copies, int_t input_size, T* input, T* target){  //only visible within this translation unit
   std::memcpy(target, input, input_size * sizeof(T));
   int_t already_copied = 1;

   while (already_copied * 2 <= no_of_copies){
      std::memcpy(target + already_copied * input_size, target, already_copied * input_size * sizeof(T));
      already_copied *= 2;
   }

   if (already_copied < no_of_copies){
      std::memcpy(target + already_copied * input_size, target, (no_of_copies - already_copied) * input_size * sizeof(T));
   }
}
}



template<typename T>
std::vector<T> UTILITY::Unidirectional_broadcast(const T* original_data, const std::vector<size_t>& original_shape, const std::vector<size_t>& target_shape, std::string original_name)
{

      std::vector<size_t> current_shape(original_shape);
      int original_length = 1;
      int target_length = 1;
      for (int i = 0; i < original_shape.size(); i++){
         original_length *= original_shape[i];
      }
      for (int i = 0; i < target_shape.size(); i++){
         target_length *= target_shape[i];
      }
      if (original_shape.size() > target_shape.size())   throw std::runtime_error("TMVA::SOFIE Error in Broadcasting Tensor " + original_name + ": original array has more dimensions than target shape ");
      auto it = current_shape.begin();
      while (current_shape.size() < target_shape.size()){
         it = current_shape.insert(it, 1);
      }

      std::vector<T> new_datavector(target_length);
      std::memcpy(new_datavector.data(), original_data, original_length * sizeof(T));

      for (int dim = target_shape.size() - 1; dim >= 0; dim--){
         if (current_shape[dim] != target_shape[dim]){
            if (current_shape[dim] != 1) throw std::runtime_error ("TMVA::SOFIE Error in Broadcasting Tensor "  + original_name + " at least one dimension to be broadcast  of the original array is not 1");

            int_t group_size = 1;
            int_t no_of_groups = 1;
            int_t no_of_copies = target_shape[dim];

            for (int i = dim + 1; i < target_shape.size(); i++){
               group_size *= current_shape[i];
            }
            for (int i = 0; i < dim; i++){
               no_of_groups *= current_shape[i];
            }

            for (int curr_group = no_of_groups - 1; curr_group >= 0; curr_group--){
               copy_vector_data<T>(no_of_copies, group_size, new_datavector.data() + curr_group * group_size,new_datavector.data() + curr_group * group_size * no_of_copies);
            }

            current_shape[dim] = target_shape[dim];
         }
      }

      return new_datavector;

}

std::vector<int_t> UTILITY::Position_to_indices(int_t position, const std::vector<size_t>& shape)
{
   std::vector<int_t> ret(shape.size(), 0);
   auto to_divide = position;
   for (int i = shape.size() - 1; i >= 0; i--){
      auto divresult = div(to_divide, shape[i]);
      ret[i] = divresult.rem;
      to_divide = divresult.quot;
   }
   return ret;
}

int_t UTILITY::Indices_to_position(const std::vector<int_t>& indices, const std::vector<size_t>& shape){
   int_t ret = 0;
   int_t rolling_length = 1;
   for (int i = indices.size() - 1; i >= 0; i--){
      ret += indices[i] * rolling_length;
      rolling_length *= shape[i];
   }
   return ret;
}





}//SOFIE
}//Experimental
}//TMVA
