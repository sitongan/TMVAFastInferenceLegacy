CXX = c++
CPPFLAGS = --std=c++11
PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`

SOFIEOBJECT = SOFIE.o ROperator_Gemm.o RDataNode.o

prototype: prototype.o sofie
	${CXX} -o prototype prototype.o $(SOFIEOBJECT) onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) ${CPPFLAGS}

prototype.o: prototype.cpp
	${CXX} ${CPPFLAGS} -c prototype.cpp

sofie: $(SOFIEOBJECT)

SOFIE.o: SOFIE.hxx SOFIE.cxx
	${CXX} ${CPPFLAGS} -c SOFIE.cxx

RDataNode.o: RDataNode.hxx RDataNode.cxx
	${CXX} ${CPPFLAGS} -c RDataNode.cxx

ROperator_Gemm.o: ROperator_Gemm.hxx ROperator_Gemm.cxx
	${CXX} ${CPPFLAGS} -c ROperator_Gemm.cxx




test: test.cpp
	${CXX} -o test test.cpp onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) ${CPPFLAGS}

clean:
	rm -f *.o
