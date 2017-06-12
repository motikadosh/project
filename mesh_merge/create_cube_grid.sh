#!/bin/bash

MESH_FILTER_BIN=../../trimesh2/bin.Linux64/mesh_filter

# -2 depth
$MESH_FILTER_BIN cube.obj -trans -2 -2 0 cube-2-20.obj
$MESH_FILTER_BIN cube.obj -trans  0 -2 0 cube0-20.obj
$MESH_FILTER_BIN cube.obj -trans  2 -2 0 cube2-20.obj

# 0 depth
$MESH_FILTER_BIN cube.obj -trans -2  0 0 cube-200.obj
$MESH_FILTER_BIN cube.obj -trans  0  0 0 cube000.obj
$MESH_FILTER_BIN cube.obj -trans  2  0 0 cube200.obj

# 2 depth
$MESH_FILTER_BIN cube.obj -trans -2  2 0 cube-220.obj
$MESH_FILTER_BIN cube.obj -trans  0  2 0 cube020.obj
$MESH_FILTER_BIN cube.obj -trans  2  2 0 cube220.obj


./mesh_merge cube-2-20.obj cube0-20.obj cube2-20.obj cube-200.obj cube000.obj cube200.obj cube-220.obj cube020.obj cube220.obj

# Lift the entire grid to z=0
$MESH_FILTER_BIN output.obj -trans  0  0 0.5 cube_grid.obj


