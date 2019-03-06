PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`
make: OpenGraph.cpp
	c++ -o OpenGraph OpenGraph.cpp onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) --std=c++11
