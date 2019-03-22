CXX=c++
CPPFLAGS=--std=c++11
PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`


prototype: prototype.o SOFIE.o
	${CXX} -o prototype prototype.o SOFIE.o onnx.pb.cc  $(ROOTCONFIG) $(PROTOBUF) ${CPPFLAGS}

prototype.o: prototype.cpp
	${CXX} ${CPPFLAGS} -c prototype.cpp

sofie: SOFIE.o

SOFIE.o: SOFIE.hxx SOFIE.cxx
	${CXX} ${CPPFLAGS} -c SOFIE.cxx

test: test.cpp
	${CXX} -o test test.cpp onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) ${CPPFLAGS}

clean:
	rm -f *.o
