#ifndef TMVA_SOFIE_RMODEL
#define TMVA_SOFIE_RMODEL

#include "SOFIE_common.hxx"
#include "RGraph.hxx"
#include <unordered_map>
#include <vector>

#include "onnx.pb.h"
#include <algorithm>
#include <fstream>
#include <ios>
#include <string>



namespace TMVA{
namespace Experimental{
namespace SOFIE{

   class RModel{
   private:
      RGraph* fGraph;
      std::unordered_map<std::string, std::string> fMetadata;
      //expected keys: "ir_version", "opset_list" (comma-separated list), "model_version", "producer_name", "producer_version", "domain", "doc_string", "model_author", "model_license")
      std::vector<int> fOpsetList;

   public:

      RModel(std::string filename){
         auto extension = filename.substr(filename.length() - 4);
         std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
         if (extension == "onnx"){


            GOOGLE_PROTOBUF_VERIFY_VERSION;
            //model I/O
            onnx::ModelProto model;
            std::fstream input(filename, std::ios::in | std::ios::binary);
            if (!model.ParseFromIstream(&input)){
               throw std::runtime_error("TMVA::SOFIE - Failed to parse onnx file");
            }

            fGraph = new RGraph (model.mutable_graph());
            fMetadata["ir_version"] = std::to_string(model.ir_version());
            fMetadata["model_version"] = std::to_string(model.model_version());

            for (int i =0; i < model.opset_import_size(); i++){
               fOpsetList.push_back(model.opset_import(i).version());
            }

            google::protobuf::ShutdownProtobufLibrary();
            std::cout << "TMVA::SOFIE - Successfully read ONNX model file " << filename << std::endl;
         }
      }

      const RGraph& GetGraph() const  {return *fGraph;}
      RGraph* GetMutableGraph() {return fGraph;}

      ~RModel(){
         delete fGraph;
      }

   };

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL
