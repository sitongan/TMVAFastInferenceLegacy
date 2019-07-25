#include <iostream>
#include <vector>
#include <string>

#include "TMVA/RTensor.hxx"
#include "TInterpreter.h"




using namespace std;



void testJitting(){

   string tojit = "\
   class DataNode{\
   public:\
   \
      string fName;\
   \
      void helloworld(){ \
         cout << \"This message is printed by jitting\" << endl; \
      }\
   \
   };\
   ";

   string tojitexec ="\
   typedef DataNode DataNode_new;\
   DataNode_new* dnode = new DataNode_new();\
   dnode->helloworld();\
   ";

   gInterpreter->Declare(tojit.c_str());
   gInterpreter->Calc(tojitexec.c_str());
}



 
///
#include <iostream>
#include <type_traits>

class foo;
class bar;

template<class T>
struct is_bar
{
    template<class Q = T>
    typename std::enable_if<std::is_same<Q, int>::value, T>::type check()
    {
        return 11;
    }

    template<class Q = T>
    typename std::enable_if<std::is_same<Q, float>::value, T>::type check()
    {
        return 11.1;
    }


};

int test_sfinae()
{
    is_bar<int> bar_int;
    is_bar<float> bar_float;
    cout << bar_int.check() << endl;
    cout << bar_float.check() << endl;

    return 0;
}


int main(){
   //testJitting();
   //test_sfinae();
   return 0;
}
