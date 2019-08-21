#ifndef TMVA_SOFIE_RGRAPH
#define TMVA_SOFIE_RGRAPH

#include "SOFIE_common.hxx"
#include "RDataNode.hxx"
#include "ROperator.hxx"

#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{



class RGraph{
private:

   std::unordered_map<std::string, int_t> fDimensionDenotation;
   std::unordered_map<std::string, RDataNodeBase*> fDataNodeMap;

   std::unordered_set<std::string> fInputDataNodeNames;

   std::vector<ROperator*> fOperatorNode;

   ROperator* make_ROperator(const onnx::NodeProto& nodeproto){
      auto find = INTERNAL::mapOptypeOperator.find(nodeproto.op_type());
      if (find == INTERNAL::mapOptypeOperator.end()){
         throw std::runtime_error("TMVA::SOFIE - Operator type " + nodeproto.op_type() + " is not yet supported");
      }else{
         return (find->second)(nodeproto, *(this));
      }
   }


public:

   RGraph(onnx::GraphProto* fONNXGraph, const std::unordered_map<std::string, int_t>& dimensionDenotationMap = {} ) :
   fDimensionDenotation(dimensionDenotationMap) {

      std::unordered_set<std::string> initializer_names;
      for (int i=0; i < fONNXGraph->initializer_size(); i++){
         initializer_names.insert(fONNXGraph->initializer(i).name());
         switch(static_cast<ETensorType>(fONNXGraph->initializer(i).data_type())){
            case ETensorType::FLOAT : {
               fDataNodeMap[fONNXGraph->initializer(i).name()] = new RDataNode<float>(fONNXGraph->mutable_initializer(i));
               break;
            }
            default: throw std::runtime_error("Data type in weight tensor " + fONNXGraph->initializer(i).name() + " not supported!\n");
         }
      }

      for (int i=0; i < fONNXGraph->input_size(); i++){
         if (initializer_names.find(fONNXGraph->input(i).name()) == initializer_names.end()){
            //input datanode is not a weight node (has no initializer)
            std::string input_name = fONNXGraph->input(i).name();
            fInputDataNodeNames.insert(input_name);

            switch(static_cast<ETensorType>(fONNXGraph->input(i).type().tensor_type().elem_type())){
               case ETensorType::FLOAT : {
                  fDataNodeMap[input_name] = new RDataNode<float>(fONNXGraph->input(i), fDimensionDenotation);
                  break;
               }
               default: throw std::runtime_error("Data type in input tensor " + input_name + " not supported!\n");
            }

         }
      }

      //ROperator* test = make_ROperator(fONNXGraph->node(0));

      for (int i=0; i < fONNXGraph->node_size(); i++){
         fOperatorNode.push_back(make_ROperator(fONNXGraph->node(i)));
      }

   }

   RGraph(){}  //external constructor

   ~RGraph(){
      for (auto const& x : fDataNodeMap){
         delete x.second;
      }
      for (auto const& x: fOperatorNode){
         delete x;
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

   void RegisterNewRDataNode(RDataNodeBase* datanode){
      std::string nodename = datanode->GetName();
      if (fDataNodeMap.find(nodename) == fDataNodeMap.end()){
         fDataNodeMap[nodename] = datanode;
      }else{
         throw std::runtime_error("Datanode with name " + nodename + " already exist\n");
      }
   }


   void Forward(){

      for (int i=0; i < fOperatorNode.size(); i++){
         //fOperatorNode[i]->Forward_reference();
         fOperatorNode[i]->Forward_blas();
      }

   }

};


}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RGRAPH
