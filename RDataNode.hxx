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
/*   protected:
   RDataNodeBase(){}*/
public:
   virtual const ETensorType& GetType() const =0;
   virtual ~RDataNodeBase(){}

};

namespace UTILITY{
template<typename T>
std::vector<T> Unidirectional_broadcast(const T* original_data, const std::vector<int_t>& original_shape, const std::vector<int_t>& target_shape, std::string original_name = "");
std::vector<int_t> Position_to_indices(int_t position, const std::vector<int_t>& shape);
int_t Indices_to_position(const std::vector<int_t>& indices, const std::vector<int_t>& shape);
}//UTILITY




template <typename T>
class RDataNode : public RDataNodeBase{

private:
   ETensorType fType;
   std::string fName = "";
   std::vector<int_t> fShape;
   int_t fLength;
   bool fHasData = false;
   std::vector<T>* fDataVector;
   bool fHasImmutableData = false;
   const T* fImmutableData;
   bool fIsSegment = false;
   std::tuple<int_t, int_t> fSegmentIndex;






   template<class Q = T>
   typename std::enable_if<std::is_same<Q, float>::value, const T*>::type get_protobuf_datafield(const onnx::TensorProto& tensorproto)
   {
       return tensorproto.float_data().data();
   }

   template<class Q = T>
   typename std::enable_if<std::is_same<Q, float>::value, void>::type set_fType()
   {
       fType = ETensorType::FLOAT;
   }

   RDataNode<T>(){};

public:
   ~RDataNode();
   RDataNode(const onnx::TensorProto& tensorproto);
   RDataNode(const std::vector<int_t>& shape, const std::string& name);
   RDataNode(const onnx::ValueInfoProto& valueinfoproto, const std::unordered_map<std::string, int_t>& dimensionDenotationMap);
   RDataNode(const std::vector<T>& input, const std::vector<int_t>& shape, const std::string& name);
   RDataNode(std::vector<T>&& input, const std::vector<int_t>& shape, const std::string& name);

   void Update(std::vector<T>&& newDataVector, const std::vector<int_t>& newShape);  //move update
   void Update(const std::vector<T>& newDataVector, const std::vector<int_t>& newShape);  //copy update

   const T* GetData();
   T* GetWriteTarget();
   const std::vector<int_t>& GetShape() const {return fShape;}
   int_t GetShape(int_t dim) const{
      //if (dim >= fShape.size() || dim < 0) throw std::runtime_error("TMVA::SOFIE - dimension requested does not exist");
      return fShape[dim];
   }
   const int_t GetLength() const {return fLength;}
   const ETensorType& GetType() const{return fType;}
   const std::string& GetName() const{return fName;};

   void Unidirectional_broadcast(const std::vector<int_t>& target_shape){
      this->Update(UTILITY::Unidirectional_broadcast(this->GetData(), this->GetShape(), target_shape, this->GetName()), target_shape);
   }

};


}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RDATANODE
