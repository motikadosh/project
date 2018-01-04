CXX = g++
CC = gcc
LD = g++
STRIP = strip
CCACHE_EXT ?= ccache

ifeq ($(LONG_MAKE),)
    PSILENT = @
else
    PSILENT =
endif

ARCH = $(shell uname -m)
ifeq ($(ARCH),x86_64)
    ARCH_LIB_DIR ?= /usr/lib/x86_64-linux-gnu
else
    ARCH_LIB_DIR ?= /usr/lib/i386-linux-gnu
endif

OPENCV_WORLD ?= 0
ifeq ($(OPENCV_WORLD),1)
    OPENCV_LFLAGS = -lopencv_world \
                    -lopencv_dnn
    OPENCV_INC = /usr/include/opencv-3.1.0/
else
    OPENCV_LFLAGS = -lopencv_core \
                    -lopencv_highgui \
                    -lopencv_imgproc \
                    -lopencv_objdetect \
                    -lopencv_calib3d \
                    -lopencv_video \
                    -lopencv_ml \
                    -lopencv_imgcodecs \
                    -lopencv_photo
    #OPENCV_INC = /usr/include
    OPENCV_INC = /usr/local/include
endif

TRIMESH_DIR = ../trimesh2
TRIMESH_LIB = $(TRIMESH_DIR)/lib.Linux64

INC_DIR ?= /usr/include
C_INC = \
    -I. \
    -I$(TOPDIR)/include \
    -I$(TOPDIR)/common \
    -I$(OPENCV_INC) \
    -I$(TOPDIR)/external/cereal/include \
    -I$(INC_DIR)/eigen3 \
    -I$(INC_DIR)/glib-2.0 \
    -I$(INC_DIR)/cairo \
    -I$(INC_DIR)/pango-1.0 \
    -I$(INC_DIR)/gdk-pixbuf-2.0 \
    -I$(INC_DIR)/atk-1.0 \
    -I$(ARCH_LIB_DIR)/glib-2.0/include \
    -I$(TRIMESH_DIR)/include

C_FLAGS = -Wall -Wno-unknown-pragmas -Wno-unused-result #-Werror

C_FLAGS += -Og -g

# Enable various supported instruction sets
SSE_EXISTS = $(shell grep -w sse /proc/cpuinfo | head -1 | wc -l)
SSE2_EXISTS = $(shell grep -w sse2 /proc/cpuinfo | head -1 | wc -l)
SSSE3_EXISTS = $(shell grep -w ssse3 /proc/cpuinfo | head -1 | wc -l)
SSE4_1_EXISTS = $(shell grep -w sse4_1 /proc/cpuinfo | head -1 | wc -l)
SSE4_2_EXISTS = $(shell grep -w sse4_2 /proc/cpuinfo | head -1 | wc -l)
AVX_EXISTS = $(shell grep -w avx /proc/cpuinfo | head -1 | wc -l)
AVX2_EXISTS = $(shell grep -w avx2 /proc/cpuinfo | head -1 | wc -l)

ifeq ($(SSE_EXISTS),1)
    C_FLAGS += -msse
endif
ifeq ($(SSE2_EXISTS),1)
    C_FLAGS += -msse2
endif
ifeq ($(SSSE3_EXISTS),1)
    C_FLAGS += -mssse3
endif
ifeq ($(SSE4_1_EXISTS),1)
    C_FLAGS += -msse4.1
endif
ifeq ($(SSE4_2_EXISTS),1)
    C_FLAGS += -msse4.2
endif
ifeq ($(AVX_EXISTS),1)
    C_FLAGS += -mavx
endif
ifeq ($(AVX2_EXISTS),1)
    C_FLAGS += -mavx2
endif

C_FLAGS += -DBOOST_ALL_DYN_LINK -DBOOST_LOG_USE_NATIVE_SYSLOG \
           -DEIGEN_MPL2_ONLY -DGLM_FORCE_RADIANS

CXX_FLAGS = $(C_FLAGS) -std=c++11

L_FLAGS = $(OPENCV_LFLAGS) \
    -lpthread \
    -lX11 \
    -lboost_serialization \
    -lboost_filesystem \
    -lboost_system \
    -lboost_date_time \
    -lboost_log \
    -lboost_log_setup \
    -lboost_thread \
    -lglog \
    -lgflags \
    -lGL -lGLU \
    -L$(TRIMESH_LIB) -ltrimesh -lgluit -fopenmp

# For non-standard install path of opencv
OPENCV_DIR := /usr/local
ifneq ($(OPENCV_DIR),)
    C_INC += -I$(OPENCV_DIR)/include
    L_FLAGS += -L$(OPENCV_DIR)/lib -Wl,--rpath $(OPENCV_DIR)/lib
endif

