#include <iostream>
#include <memory>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

#include "TriMesh.h"

#define DBG(params) \
    std::cout << dbgCount++ << ") " << __FUNCTION__ << ": " << params << std::endl

std::shared_ptr<trimesh::TriMesh> themesh, themeshMerge;
int dbgCount = 0;

template <typename T>
inline std::ostream& operator<<(std::ostream &out, const std::vector<T> &vector)
{
    for(auto i : vector)
        out << i << ", ";

    if (vector.size())
        out << "\b\b";

    return out;
}

void MergeMesh(std::shared_ptr<trimesh::TriMesh> &themesh, const std::shared_ptr<trimesh::TriMesh> &themeshMerge)
{
    int offset = themesh->vertices.size();

    for (auto v : themeshMerge->vertices)
        themesh->vertices.push_back(v);

    for (auto f : themeshMerge->faces)
    {
        trimesh::TriMesh::Face newF(f[0] + offset, f[1] + offset, f[2] + offset);
        themesh->faces.push_back(newF);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Input should have at least 2 mesh files" << std::endl;
        exit(1);
    }

    std::string baseMeshFile = argv[1];

    std::vector<std::string> mergeFiles;
    for (int i = 2; i < argc; i++)
        mergeFiles.push_back(argv[i]);

    DBG("Input files: " << baseMeshFile << ", " << mergeFiles);

    themesh = std::shared_ptr<trimesh::TriMesh>(trimesh::TriMesh::read(baseMeshFile));
    if (!themesh)
    {
        std::cerr << "Failed reading model file: " << baseMeshFile << std::endl;
        exit(1);
    }

    for (auto file : mergeFiles)
    {
        themeshMerge = std::shared_ptr<trimesh::TriMesh>(trimesh::TriMesh::read(file));
        if (!themeshMerge)
        {
            std::cerr << "Failed reading model file: " << file << std::endl;
            exit(1);
        }

        MergeMesh(themesh, themeshMerge);
    }

    DBG("Vertices num [" << themesh->vertices.size() << "], faces num [" << themesh->faces.size() << "]");
    themesh->write("output.obj");

    return 0;
}

