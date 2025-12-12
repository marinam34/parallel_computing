ARCH      ?= sm_60
HOST_COMP ?= mpicc 

NVCC      := nvcc
CXX       := $(HOST_COMP)

INC_DIR   := include
SRC_DIR   := src
BUILD_DIR := build

CXXFLAGS  := -O3 -std=c++11 -I$(INC_DIR)
NVCCFLAGS := -O3 -std=c++11 -arch=$(ARCH) -ccbin $(HOST_COMP) -I$(INC_DIR)

OBJS = \
  $(BUILD_DIR)/config.o \
  $(BUILD_DIR)/analytic.o \
  $(BUILD_DIR)/grid_mpi.o \
  $(BUILD_DIR)/main_cuda.o \
  $(BUILD_DIR)/solver_cuda.o

all: $(BUILD_DIR)/task_cuda

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/config.o: $(SRC_DIR)/config.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/analytic.o: $(SRC_DIR)/analytic.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/grid_mpi.o: $(SRC_DIR)/grid_mpi.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main_cuda.o: $(SRC_DIR)/main_cuda.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/solver_cuda.o: $(SRC_DIR)/solver_cuda.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/task_cuda: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm


clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
