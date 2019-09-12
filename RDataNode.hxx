#ifndef TMVA_SOFIE_RDATANODE
#define TMVA_SOFIE_RDATANODE

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

#include "SOFIE_common.hxx"


#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


class RDataNodeBase{
protected:
   ETensorType fType;
   std::string fName = "";
   std::vector<std::size_t> fShape;
   int_t fLength;
   bool fIsSegment = false;
   std::tuple<int_t, int_t> fSegmentIndex;

public:

   const ETensorType& GetType() const{return fType;}
   virtual ~RDataNodeBase(){}
   RDataNodeBase(std::string name): fName(name){}
   RDataNodeBase(){}
   const std::string& GetName() const{return fName;}
   void SetName(const std::string& newName) {fName = newName;}
   const std::vector<std::size_t>& GetShape() const {return fShape;}
   int_t GetShape(int_t dim) const{
      //if (dim >= fShape.size() || dim < 0) throw std::runtime_error("TMVA::SOFIE - dimension requested does not exist");
      return fShape[dim];
   }
   const int_t GetLength() const {return fLength;}

   bool HasSameShape(const RDataNodeBase& another){
      auto this_shape = this->GetShape();
      auto another_shape = another.GetShape();
      if (this_shape.size() != another_shape.size()) return false;
      for (int dim = 0; dim < this_shape.size(); dim++){
         if (this_shape[dim] != another_shape[dim]) return false;
      }
      return true;
   }


};

namespace UTILITY{
template<typename T>
std::vector<T> Unidirectional_broadcast(const T* original_data, const std::vector<size_t>& original_shape, const std::vector<size_t>& target_shape, std::string original_name = "");
std::vector<int_t> Position_to_indices(int_t position, const std::vector<size_t>& shape);
int_t Indices_to_position(const std::vector<int_t>& indices, const std::vector<size_t>& shape);
}//UTILITY




template <typename T>   //expecting T to be of same interface as RTensor
class RDataNode : public RDataNodeBase{

private:


   //std::vector<T>* fDataVector;
   T* fDataTensor;


   typename std::enable_if<std::is_same<typename T::Value_t, float>::value, void>::type extract_protobuf_datafield(onnx::TensorProto* tensorproto)
   {
      //fDataVector = new std::vector<T>(tensorproto->float_data_size());
      tensorproto->mutable_float_data()->ExtractSubrange(0, tensorproto->float_data_size(), fDataTensor->GetData());
      //this is a copy
   }

   typename std::enable_if<std::is_same<typename T::Value_t, float>::value, void>::type set_fType()
   {
       fType = ETensorType::FLOAT;
   }


   //RDataNode<T>(){};

public:

   using Value_t = typename T::Value_t;
   using Container_t = typename T::Container_t;

   ~RDataNode();
   RDataNode(onnx::TensorProto* tensorproto);
   RDataNode(const std::vector<std::size_t>& shape, const std::string& name);
   RDataNode(const onnx::ValueInfoProto& valueinfoproto, const std::unordered_map<std::string, int_t>& dimensionDenotationMap);
   RDataNode(const std::vector<T>& input, const std::vector<size_t>& shape, const std::string& name);
   RDataNode(Container_t&& input, const std::vector<size_t>& shape, const std::string& name);

   void Update(Container_t&& newDataVector, const std::vector<size_t>& newShape);  //move update
   void Update(const std::vector<T>& newDataVector, const std::vector<size_t>& newShape);  //copy update
   RDataNode<T>& operator=(const RDataNode<T>& other);   //copy assignment


   const Value_t* GetData() const;
   Value_t* GetData();






   void Unidirectional_broadcast(const std::vector<size_t>& target_shape){
      this->Update(UTILITY::Unidirectional_broadcast(this->GetData(), this->GetShape(), target_shape, this->GetName()), target_shape);
   }

};


}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RDATANODE
