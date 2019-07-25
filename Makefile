CXX = g++
CPPFLAGS = --std=c++11
PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`

SOFIEOBEJCT = RDataNode.o ROperator_Gemm.o ROperator_Transpose.o ROperator_Relu.o
SOFIEHEADER = RGraph.hxx ROperator.hxx ROperator_Gemm.hxx ROperator_Transpose.hxx ROperator_Relu.hxx
SOFIE = $(SOFIEOBEJCT) $(SOFIEHEADER)

prototype: prototype.o $(SOFIE)
	${CXX} -o prototype prototype.o $(SOFIEOBEJCT) onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) ${CPPFLAGS}

prototype.o: prototype.cpp
	${CXX} ${CPPFLAGS} -c prototype.cpp

$(filter %.o, $(SOFIEOBEJCT)): %.o: %.cxx
	${CXX} ${CPPFLAGS} -c $<



test: test.cpp
	${CXX} ${CPPFLAGS} -Wall -g $(ROOTCONFIG) $(PROTOBUF) -o test test.cpp onnx.pb.cc

.phony: clean
clean:
	rm *.o
