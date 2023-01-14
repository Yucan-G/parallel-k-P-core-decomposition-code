# Parallel (k, P)-core decomposition

Parallel (k,P)-core decomposition algorithms over large HINs

* BasicAPCore_FastAPCore.cpp: Including the source code of HomBCore, BasicAPCore, FastAPCore, and sparse matrix related functions
* BasicAPCore_FastAPCore_for_KGs.cpp: Including the source code of BasicAPCore and FastAPCore with optimizations for KGs
* BoolAPCore.cpp: Including the source code of BoolAPCore and the sparse boolean matrix related functions

## Compiling the program

```
$g++ -fopenmp -o BasicAPCore_FastAPCore BasicAPCore_FastAPCore.cpp
$g++ -fopenmp -o BasicAPCore_FastAPCore_for_KGs BasicAPCore_FastAPCore_for_KGs.cpp
$g++ -fopenmp -o BoolAPCore BoolAPCore.cpp
```

## Input format

* HIN file

  An HIN with vertex start from 0, and edge type start form 1. The first line of the file should contain 3 integers: number of vertices, number of edges, and number of edge types.
* meta-path file

  A meta-path consists of edge types, in which the negative numbers denote reverse edge types. The first line of the file is the length of the meta-path.
* file path
  File paths can be changed at the begining of the source code

  ```
  /*path of input files*/
  char* HIN_PATH = (char*)"Movielens/graph_Movielens.txt";
  char* METAPATH_PATH = (char*)"Movielens/metapath.txt";
  ```
