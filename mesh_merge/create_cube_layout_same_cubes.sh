#!/bin/bash

MESH_FILTER_BIN=../../trimesh2/bin.Linux64/mesh_filter

# -2 depth
$MESH_FILTER_BIN cube.obj -trans -2 -3 0 cube-2-30.obj
$MESH_FILTER_BIN cube.obj -trans  0 -3 0 cube0-30.obj
$MESH_FILTER_BIN cube.obj -trans  2 -3 0 cube2-30.obj

# 0 depth
$MESH_FILTER_BIN cube.obj -trans -1  0 0 cube-100.obj
$MESH_FILTER_BIN cube.obj -trans  0  0 0 cube000.obj
$MESH_FILTER_BIN cube.obj -trans  3  0 0 cube300.obj
$MESH_FILTER_BIN cube.obj -trans  4  0 0 cube400.obj

# 2 depth
$MESH_FILTER_BIN cube.obj -trans -2  2 0 cube-220.obj
$MESH_FILTER_BIN cube.obj -trans  2  2 0 cube220.obj


./mesh_merge cube-2-30.obj cube0-30.obj cube2-30.obj cube-100.obj cube000.obj cube300.obj cube400.obj cube-220.obj cube220.obj

# Lift the entire grid to z=0
$MESH_FILTER_BIN output.obj -trans  0  0 0.5 cube_layout.obj


