# --- Compiler and Tools ---
NVCC      := nvcc
CXX       := g++
EXE       := bin/cuda_project

# --- Directories ---
SRC_DIR   := src
OBJ_DIR   := obj
BIN_DIR   := bin
INC_DIR   := include

# --- Compilation Flags ---
# -O3: Optimization, -std=c++11: Language standard
CXXFLAGS  := -O3 -std=c++11 -I$(INC_DIR)
# -arch: Target GPU architecture (e.g., sm_70, sm_80)
# -Xcompiler: Pass flags to the underlying host compiler
NVCCFLAGS := -O3 -std=c++11 -I$(INC_DIR) -arch=sm_80 -Xcompiler -fPIC

# --- Library Links ---
# -lcudart: Essential CUDA runtime library
LIBS      := -lcudart

# --- Source and Object Files ---
CPP_SRCS  := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu)
OBJS      := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS)) \
             $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.cu.o, $(CU_SRCS))

# --- Build Rules ---
all: $(BIN_DIR) $(OBJ_DIR) $(EXE)

# Final Linking: nvcc is used as the linker to handle both host and device objects
$(EXE): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LIBS)

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Create directories if they don't exist
$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean