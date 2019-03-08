PROTOBUF = `pkg-config --cflags --libs protobuf`
ROOTCONFIG = `root-config --cflags --glibs`
make: sofie.cpp
	c++ -o sofie sofie.cpp onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) --std=c++11
test: test.cpp
	c++ -o test test.cpp onnx.pb.cc $(ROOTCONFIG) $(PROTOBUF) --std=c++11
