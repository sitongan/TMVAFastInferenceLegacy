#ifndef TMVA_SOFIE
#define TMVA_SOFIE

#include <cstddef>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <array>
#include <unordered_set>
#include <queue>
#include <deque>
#include <cstdint> //int64_t: used for ONNX IR
#include <cstddef> // std::size_t
#include <cassert>

#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{   //Project code. System for Open, Fast Inference and Evaluation

   typedef std::int64_t int_t;   //int64 default int used in onnx.pb.h
   typedef float float64_t;

   enum class EAttributeType{
      UNDEFINED, FLOAT, INT, STRING, TENSOR, GRAPH, FLOATS, INTS, STRINGS, TENSORS, GRAPHS   //order sensitive
   };
   //Supported: NONE
   enum class ETensorType{
      UNDEFINED, FLOAT, UNINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL, FLOAT16, DOUBLE,  //order sensitive
      UINT32, UINT64, COMPLEX64, COMPLEX28, BFLOAT16
   };
   //Supported:FLOAT

   template <typename T>
   class RDataNode{
      std::string fName = "";
      std::vector<int_t> fShape;
      //ETensorType fType;
      bool fIsSegment;
      std::tuple<int_t, int_t> fSegmentIndex;
      bool fHasData = false;
      bool fHasImmutableData = false;
      int_t fLength;
      std::vector<T>* fDataVector;
      const T* fImmutableData;

   public:
      ~RDataNode();
      RDataNode(const onnx::TensorProto& tensorproto);
      RDataNode(const onnx::ValueInfoProto& valueinfoproto);
      RDataNode(const T* data, const std::vector<int_t>& shape, const std::string& name = "");
      const T* GetData();
      const std::vector<int_t>& GetShape(){return fShape;}

   private:
      template<class Q = T>
      typename std::enable_if<std::is_same<Q, float64_t>::value, const T*>::type get_protobuf_datafield(const onnx::TensorProto& tensorproto)
      {
          return fImmutableData = tensorproto.float_data().data();
      }
   };



   template <typename T>
   SOFIE::RDataNode<T>::RDataNode(const onnx::TensorProto& tensorproto)
   {
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
         //const float64_t* raw_data = reinterpret_cast<const float64_t*>(tensorproto.raw_data().c_str());
         //fDataVector = new std::vector<float64_t>(raw_data, raw_data+fLength);
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
   SOFIE::RDataNode<T>::~RDataNode(){
      if (fHasData){
         delete fDataVector;
      }
   }

   template <typename T>
   const T* SOFIE::RDataNode<T>::GetData(){
      if (fHasImmutableData){
         return fImmutableData;
      }else if (fHasData){
         return fDataVector->data();
      }else{
         throw std::runtime_error("Tensor " + fName + " has no data.");
      }
   }


   class RAttribute{ // interface for attributes
   public:
      template<typename T>
      T& GetValue(){
            //return (RAttribute<T> *)(this)->fValue;
      }
      // in the operator functions:
      //      Gemm(const RAttribute& att1){
      //         att1.GetValue<int>
      //      }
   protected:
      std::string fName;
      std::size_t fSize;
      EAttributeType fType;

   };

   template<typename T>
   class RAttributeData: public RAttribute{
         T fValue;
   };
   //template specialisation for list of attributes?

   class ROnnxOperator{
   public:

   };

   class ROpNode{
   public:


      //temp
      const onnx::NodeProto& fNodeProto; //for data access

      ROpNode(const onnx::NodeProto& nodeproto): fNodeProto(nodeproto) {}

   //private:
      //std::vector<RDataNode*> fWeights;
      //std::vector<RDataNode*> fInputs;
      //std::vector<RDataNode*> fOutputs;
      std::vector<RAttribute*> fAttributes;
      std::string fOperator;
      //todo: map from string to function pointers for the operator
   };


   class RGraph{
   public:

   //private:
      std::string fName;
      std::map<int_t, std::vector<int_t>> fEdgesForward;
      std::map<int_t, std::vector<int_t>> fEdgesBackward;
      //std::vector<RDataNode> fDataNodes;
      std::vector<ROpNode> fOpNodes;
   };


   class RModel{
   public:
   private:
      int64_t fIRVersion;
      std::vector<std::tuple<std::string, int64_t>> fOpsetVersion;
      int64_t fModelVersion;
      std::map<std::string, std::string> fMetadata;
         //expected keys: "producer_name", "producer_version", "domain", "doc_string", "model_author", "model_license"
      RGraph fGraph; //main graph to be executed for the model
   };

void check_init_assert();



std::string print_nodelist(const std::unordered_set<int_t>& vec, const onnx::GraphProto& graph);

std::vector<int_t> topological_sort(const std::map<int_t, std::unordered_set<int_t>>& EdgesForward,
                                    const std::map<int_t, std::unordered_set<int_t>>& EdgesBackward);

}
}
}

#endif //TMVA_SOFIE
