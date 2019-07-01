#ifndef TMVA_SOFIE_RDATANODE
#define TMVA_SOFIE_RDATANODE

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "SOFIE.hxx"
#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


      class RDataNodeBase{
   /*   protected:
         RDataNodeBase(){}*/
      public:
         virtual const ETensorType& GetType() const =0;
         virtual ~RDataNodeBase(){}
      };


      template <typename T>
      class RDataNode : public RDataNodeBase{

      private:
         std::string fName = "";
         std::vector<int_t> fShape;
         ETensorType fType;
         bool fIsSegment = false;
         std::tuple<int_t, int_t> fSegmentIndex;
         bool fHasData = false;
         bool fHasImmutableData = false;
         int_t fLength;
         std::vector<T>* fDataVector;



         const T* fImmutableData;

         template<class Q = T>
         typename std::enable_if<std::is_same<Q, float>::value, const T*>::type get_protobuf_datafield(const onnx::TensorProto& tensorproto)
         {
             return fImmutableData = tensorproto.float_data().data();
         }

         template<class Q = T>
         typename std::enable_if<std::is_same<Q, float>::value, void>::type set_fType()
         {
             fType = ETensorType::FLOAT;
         }

         RDataNode<T>(){};

      public:
         ~RDataNode();
         RDataNode<T>(const onnx::TensorProto& tensorproto);
         RDataNode<T>(const std::vector<int_t>& shape, const std::string& name);
         RDataNode<T>(const onnx::ValueInfoProto& valueinfoproto, const std::unordered_map<std::string, int_t>& dimensionDenotationMap);
         RDataNode<T>(const std::vector<T>& input, const std::vector<int_t>& shape, const std::string& name);
         void Update(std::vector<T>& newDataVector, std::vector<int_t> newShape);
         const T* GetData();
         const std::vector<int_t>& GetShape() const {return fShape;}
         const int_t GetLength() const {return fLength;}
         const ETensorType& GetType() const{return fType;}
         const std::string& GetName() const{return fName;};

      };


namespace INTERNAL{
template<typename T>
void copy_vector_data(int_t no_of_copies, int_t input_size, T* input, T* target){
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
}//INTERNAL



namespace UTILITY{
template<typename T>
void unidirectional_broadcast(RDataNode<T>& original, const std::vector<int_t>& target_shape){

      const std::vector<int_t>& original_shape = original.GetShape();
      std::vector<int_t> current_shape(original_shape);
      if (original_shape.size() > target_shape.size())   throw std::runtime_error("TMVA::SOFIE Error in Broadcasting Tensor " + original.GetName() + ": original array has more dimensions than target shape ");
      auto it = current_shape.begin();
      while (current_shape.size() < target_shape.size()){
         it = current_shape.insert(it, 1);
      }

      int target_length = 1;
      for (int i = 0; i < target_shape.size(); i++){
         target_length *= target_shape[i];
      }

      std::vector<T> new_datavector(target_length);
      std::memcpy(new_datavector.data(), original.GetData(), original.GetLength() * sizeof(T));

      for (int dim = target_shape.size() - 1; dim >= 0; dim--){
         if (current_shape[dim] != target_shape[dim]){
            if (current_shape[dim] != 1) throw std::runtime_error ("TMVA::SOFIE Error in Broadcasting Tensor "  + original.GetName() + " at least one dimension to be broadcast  of the original array is not 1");

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
               INTERNAL::copy_vector_data<T>(no_of_copies, group_size, new_datavector.data() + curr_group * group_size,new_datavector.data() + curr_group * group_size * no_of_copies);
            }

            current_shape[dim] = target_shape[dim];
         }
      }

      original.Update(new_datavector, target_shape);

}
}//UTILITY


}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RDATANODE
