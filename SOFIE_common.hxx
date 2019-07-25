#ifndef TMVA_SOFIE_COMMON
#define TMVA_SOFIE_COMMON


#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <array>
#include <unordered_set>
#include <queue>
#include <deque>
#include <cstdint> //int64_t: used for ONNX IR
#include <cstddef> // std::size_t
#include <cassert>
#include <cstdio>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

   typedef std::int64_t int_t;   //int64 default int used in onnx.pb.h

   enum class ETensorType{
      UNDEFINED = 0, FLOAT = 1, UNINT8 = 2, INT8 = 3, UINT16 = 4, INT16 = 5, INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9, //order sensitive
       FLOAT16 = 10, DOUBLE = 11, UINT32 = 12, UINT64 = 13, COMPLEX64 = 14, COMPLEX28 = 15, BFLOAT16 = 16
   };
   //Supported:FLOAT



}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE
