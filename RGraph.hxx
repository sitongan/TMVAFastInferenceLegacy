#ifndef TMVA_SOFIE_RGRAPH
#define TMVA_SOFIE_RGRAPH

#include "SOFIE.hxx"
#include "ROperator.hxx"
#include "RDataNode.hxx"
#include "ROperator_Gemm.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RGraph;
class ROperator;

class RGraph{
private:
   //std::string fName;
   //std::map<int_t, std::vector<int_t>> fEdgesForward;
   //std::map<int_t, std::vector<int_t>> fEdgesBackward;

   std::unordered_map<std::string, int_t> fDimensionDenotation;
   std::unordered_map<std::string, RDataNodeBase*> fDataNodeMap;

   std::unordered_set<std::string> fInputDataNodeNames;

   std::vector<ROperator*> fOperatorNode;

   const onnx::GraphProto& fONNXGraph;

   ROperator* make_ROperator(const onnx::NodeProto& nodeproto){
      return INTERNAL::mapOptypeOperator[nodeproto.op_type()](nodeproto, *(this));
   }

public:

   RGraph(const onnx::GraphProto& onnxGraph, const std::unordered_map<std::string, int_t>& dimensionDenotationMap = {} ) :
   fONNXGraph(onnxGraph), fDimensionDenotation(dimensionDenotationMap) {

      std::unordered_set<std::string> initializer_names;
      for (int i=0; i < fONNXGraph.initializer_size(); i++){
         initializer_names.insert(fONNXGraph.initializer(i).name());
         switch(fONNXGraph.initializer(i).data_type()){
            case onnx::TensorProto::FLOAT : {
               fDataNodeMap[fONNXGraph.initializer(i).name()] = new RDataNode<float>(fONNXGraph.initializer(i));
               break;
            }
            default: throw std::runtime_error("Data type in weight tensor " + fONNXGraph.initializer(i).name() + " not supported!\n");
         }
      }

      for (int i=0; i < fONNXGraph.input_size(); i++){
         if (initializer_names.find(fONNXGraph.input(i).name()) == initializer_names.end()){
            //input datanode is not a weight node (has no initializer)
            std::string input_name = fONNXGraph.input(i).name();
            fInputDataNodeNames.insert(input_name);

            switch(static_cast<ETensorType>(fONNXGraph.input(i).type().tensor_type().elem_type())){
               case ETensorType::FLOAT : {
                  fDataNodeMap[input_name] = new RDataNode<float>(fONNXGraph.input(i), fDimensionDenotation);
                  break;
               }
               default: throw std::runtime_error("Data type in input tensor " + input_name + " not supported!\n");
            }

         }
      }

   ROperator* test = make_ROperator(fONNXGraph.node(0));
   //ROperator* test = INTERNAL::make_ROperator_Gemm(fONNXGraph.node(0), *(this));
   /*
      for (int i=0; i < fONNXGraph.node_size(); i++){
         fOperatorNode.push_back(make_ROperator(fONNXGraph.node(i)));
      }
      */
   }

   ~RGraph(){
      for (auto const& x : fDataNodeMap){
         delete x.second;
      }
   }


   RDataNodeBase* GetRDataNode(std::string name) const {
      auto datanode_queried = fDataNodeMap.find(name);
      if (datanode_queried != fDataNodeMap.end()){
         return (datanode_queried->second);
      }else{
         throw std::runtime_error("Datanode " + name + " not found!\n");
      }
   }

   void RegisterNewRDataNode(std::string name, RDataNodeBase* datanode){
      if (fDataNodeMap.find(name) == fDataNodeMap.end()){
         fDataNodeMap[name] = datanode;
      }else{
         throw std::runtime_error("Datanode with name " + name + " already exist\n");
      }
   }
};



}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RGRAPH
