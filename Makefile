PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`
make: prototype.cpp
	c++ -o prototype prototype.cpp onnx.pb.cc SOFIE.cxx $(ROOTCONFIG) $(PROTOBUF) --std=c++11
test: test.cpp
	c++ -o test test.cpp onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) --std=c++11
