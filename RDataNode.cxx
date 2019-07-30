
#include "RDataNode.hxx"
#include <cstring>
#include <cstdlib>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

//constructor for immutable tensorproto (weights)
//construct a wrapper around the underlying data
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


   if (tensorproto->has_raw_data()){

      auto raw_data_ptr = const_cast<T*>(reinterpret_cast<const T*>(tensorproto->release_raw_data()->c_str()));
      fDataVector = new std::vector<T>(raw_data_ptr, raw_data_ptr + fLength);
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
   fDataVector->resize(fLength);
}

//construct from a std vector, copy data
template <typename T>
RDataNode<T>::RDataNode(const std::vector<T>& input, const std::vector<int_t>& shape, const std::string& name) :fName(name){
   set_fType();
   fShape = shape;
   fLength = input.size();
   fDataVector = new std::vector<T>(input);
}

//construct from a r-value std vector, move data
template <typename T>
RDataNode<T>::RDataNode(std::vector<T>&& input, const std::vector<int_t>& shape, const std::string& name) :fName(name){
   set_fType();
   fShape = shape;
   fLength = input.size();
   fDataVector = new std::vector<T>(input);
}

//constructor by shape and type
template <typename T>
RDataNode<T>::RDataNode(const std::vector<int_t>& shape, const std::string& name) :fName(name){
   set_fType();
   fShape = shape;
   fLength = 1;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   fDataVector = new std::vector<T>();
   fDataVector->resize(fLength);
}




template <typename T>
RDataNode<T>::~RDataNode(){
   delete fDataVector;
}

template <typename T>
const T* RDataNode<T>::GetData(){
   return fDataVector->data();
}

template <typename T>
T* RDataNode<T>::GetMutable(){
   return fDataVector->data();
}


template <typename T>
void RDataNode<T>::Update(std::vector<T>&& newDataVector, const std::vector<int_t>& newShape){
   fLength = 1;
   fShape = newShape;   //copy assignment
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   if (fLength != newDataVector.size()) throw std::runtime_error("TMVA::SOFIE - RDataNode Update Error, size inconsistency");
   *fDataVector = newDataVector;  //this will be move assignment
}

template <typename T>
void RDataNode<T>::Update(const std::vector<T>& newDataVector,const std::vector<int_t>& newShape){
   fLength = 1;
   fShape = newShape;
   for (auto const& dim_size : fShape){
      fLength *= dim_size;
   }
   if (fLength != newDataVector.size()) throw std::runtime_error("TMVA::SOFIE - RDataNode Update Error, size inconsistency");
   *fDataVector = newDataVector;  //this will be copy assignment
}

template class RDataNode<float>;   //explicit template initialization






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



template<typename T>
std::vector<T> UTILITY::Unidirectional_broadcast(const T* original_data, const std::vector<int_t>& original_shape, const std::vector<int_t>& target_shape, std::string original_name)
{

      std::vector<int_t> current_shape(original_shape);
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

std::vector<int_t> UTILITY::Position_to_indices(int_t position, const std::vector<int_t>& shape)
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

int_t UTILITY::Indices_to_position(const std::vector<int_t>& indices, const std::vector<int_t>& shape){
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
