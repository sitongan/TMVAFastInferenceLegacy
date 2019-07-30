CXX = g++
CPPFLAGS = --std=c++11 -MMD -MP
PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`

SRC = ${wildcard *.cxx}
SOFIEOBEJCT = RDataNode.o ROperator_Gemm.o ROperator_Transpose.o ROperator_Relu.o
SOFIEHEADER = RGraph.hxx ROperator.hxx ROperator_Gemm.hxx ROperator_Transpose.hxx ROperator_Relu.hxx
SOFIE = $(SOFIEOBEJCT) $(SOFIEHEADER)

prototype: ${SRC:%.cxx=%.o}
	${CXX} -o prototype $^ $(ROOTCONFIG) $(PROTOBUF) ${CPPFLAGS}

-include $(SRC:%.cxx=%.d)

%.o: %.cxx
	${CXX} ${CPPFLAGS} -c $<

.phony: clean
clean:
	-rm *.d
	-rm *.o
