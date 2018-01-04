#!/bin/bash

MESH_FILTER_BIN=../../trimesh2/bin.Linux64/mesh_filter

# -2 depth
$MESH_FILTER_BIN square.obj -trans  0 0 0 square000.obj
#$MESH_FILTER_BIN square.obj -trans  1 0 0 square100.obj
#$MESH_FILTER_BIN square.obj -trans  0 1 0 square010.obj
$MESH_FILTER_BIN square.obj -trans  1 1 0 square110.obj

#./mesh_merge square000.obj square100.obj square010.obj square110.obj
./mesh_merge square000.obj square110.obj
mv output.obj grid2x2.obj

$MESH_FILTER_BIN grid2x2.obj -trans  0 0 0 square000.obj
$MESH_FILTER_BIN grid2x2.obj -trans  2 0 0 square200.obj
$MESH_FILTER_BIN grid2x2.obj -trans  0 2 0 square020.obj
$MESH_FILTER_BIN grid2x2.obj -trans  2 2 0 square220.obj

./mesh_merge square000.obj square200.obj square020.obj square220.obj
mv output.obj grid4x4.obj

$MESH_FILTER_BIN grid4x4.obj -trans  0 0 0 square000.obj
$MESH_FILTER_BIN grid4x4.obj -trans  4 0 0 square400.obj
$MESH_FILTER_BIN grid4x4.obj -trans  0 4 0 square040.obj
$MESH_FILTER_BIN grid4x4.obj -trans  4 4 0 square440.obj

./mesh_merge square000.obj square400.obj square040.obj square440.obj
mv output.obj grid8x8.obj

