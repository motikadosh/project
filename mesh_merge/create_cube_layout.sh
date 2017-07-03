#!/bin/bash

MESH_FILTER_BIN=../../trimesh2/bin.Linux64/mesh_filter
BASE_CUBE=cube000.obj

# Lift initial cube to z=0
$MESH_FILTER_BIN cube.obj -trans 0 0 0.5 $BASE_CUBE


# -2 depth
$MESH_FILTER_BIN $BASE_CUBE -scale 1 1 3 -trans -2 -3 0 cube-2-30.obj
$MESH_FILTER_BIN $BASE_CUBE -scale 2 2 2 -trans  0 -3 0 cube0-30.obj
$MESH_FILTER_BIN $BASE_CUBE -trans  2 -3 0 cube2-30.obj

# 0 depth
$MESH_FILTER_BIN $BASE_CUBE -scale 1 1 4 -trans -1  0 0 cube-100.obj
#$MESH_FILTER_BIN $BASE_CUBE -trans  0  0 0 cube000.obj
$MESH_FILTER_BIN $BASE_CUBE -scale 2 1 1 -trans  3  0 0 cube300.obj

# 2 depth
$MESH_FILTER_BIN $BASE_CUBE -scale 1 3 1 -trans -2  2 0 cube-220.obj
$MESH_FILTER_BIN $BASE_CUBE -scale 3 3 3 -trans  2  3 0 cube220.obj


./mesh_merge cube-2-30.obj cube0-30.obj cube2-30.obj cube-100.obj cube300.obj cube-220.obj cube220.obj

cp output.obj cube_layout.obj

