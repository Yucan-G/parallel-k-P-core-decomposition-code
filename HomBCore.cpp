/*********** Code for HomBCore ***********/

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <set>
#include <vector>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <unistd.h> 
using namespace std;

int NUM_THREADS = 1;

/*path of input files*/
char* HIN_PATH = (char*)"Movielens/graph_Movielens.txt";
char* VERTEX_PATH = (char*)"Movielens/vertices_Movielens.txt";
char* METAPATH_PATH = (char*)"Movielens/metapath_HomBCore.txt";

typedef struct {
	int v;
	int type;
}edge;
typedef struct {
	vector<edge> e;
	int type;
}vertice;

typedef struct {
	int n;
	int m;
	int t;
	vector<vertice> v;
}graph;
graph g;

typedef vector<int> neighbor;

typedef struct {
	int i;
	int j;
	int value;
}Triplet;
typedef vector <Triplet> triplet;

typedef struct {
	int j;
	int value;
}Tuple;
typedef vector <Tuple> tuples;

typedef int Index;

typedef struct {
	int row, col;
	vector<Index> lhs_index, rhs_index;
	tuples lhs_elements, rhs_elements;
}matrix;

vector<int> path;
vector<int> core;
int path_length, ini_type;
double HomBCore_runtime;

void read_env();
int load_HIN(char* filename);
int load_metapath(char* filename);
void HomBCore();
void parallel_core_decomposition(matrix Adjacent);
void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor> &neighbors);

void setFromTriplets_lhs(matrix& A, triplet t);
int nonZeros(matrix A);

int main(int argc, char* argv[])
{
	read_env();
	load_HIN(HIN_PATH);
	load_metapath(METAPATH_PATH); 

	HomBCore();
	
	return 0;
}

void read_env()
{

#pragma omp parallel
	{
#pragma omp master
		NUM_THREADS = omp_get_num_threads();
	}
	printf("NUM_PROCS:     %d \n", omp_get_num_procs());
	printf("NUM_THREADS:   %d \n", NUM_THREADS);
}

int load_HIN(char* filename)
{
	FILE* infp = fopen(filename, "r");
	if (infp == NULL) {
		fprintf(stderr, "Error: could not open input file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading HIN from: %s\n", filename);
	fscanf(infp, "%d %d %d\n", &(g.n), &(g.m), &(g.t));
	vertice tmp_v; triplet tmp_t; matrix tmp_m;
	tmp_m.row = tmp_m.col = g.n;
	for (int i = 0; i < g.n; i++) { g.v.push_back(tmp_v); core.push_back(0);}
	int u, v, type;
	edge e;
	while (fscanf(infp, "%d %d %d\n", &u, &v, &type) != EOF) {
		e.v = v; e.type = type;
		g.v[u].e.push_back(e);
		e.v = u; e.type = -type;
		g.v[v].e.push_back(e);
	}
	fclose(infp);

	infp = fopen(VERTEX_PATH, "r");
	if (infp == NULL) {
		fprintf(stderr, "Error: could not open input file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading vertices from: %s\n", filename);	
	for (int i = 0; i < g.n; i++)
	{
		fscanf(infp, "%d\n", &type);
		g.v[i].type = type;
	}
	fclose(infp);
	return 0;
}

int load_metapath(char* filename)
{
	FILE* infp = fopen(filename, "r");
	if (infp == NULL) {
		fprintf(stderr, "Error: could not open inputh file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading metapath: %s\n\n", filename);
	fscanf(infp, "%d %d\n", &path_length, &ini_type);
	int type;
	for (int i = 1; i < path_length; i++)
	{
		fscanf(infp, "%d ", &type);
		path.push_back(type);
	}
	fscanf(infp, "%d", &type);
	path.push_back(type);
	return 0;
}

void HomBCore()
{
	double run_time;
	timeval time_start;
	timeval time_over;
	timeval time1, time2;
	cout << "Ready to run HomBCore..." << endl;
	gettimeofday(&time_start,NULL);
	gettimeofday(&time1,NULL);
	matrix A1 = { g.n, g.n };
	triplet triplet1;
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < g.n; i++)
	{
		if (g.v[i].type == ini_type) //v is a vertice with the target type
		{
			vector<int> x;
			x.push_back(i);
			for (int l = 0; l < path_length; l++)
			{
				set<int> y;
				int cur_t = path[l];
				for (int k = 0; k < x.size(); k++)
				{
					int cur_v = x[k];
					for (int m = 0; m < g.v[cur_v].e.size(); m++)
						if (g.v[cur_v].e[m].type == cur_t) y.insert(g.v[cur_v].e[m].v);
				}
				x.assign(y.begin(), y.end());
			}
			set<int> s(x.begin(), x.end());
			Triplet t;
			t.i = i;
			t.value = 1;
#pragma omp critical
			{
				for (int v : s)
				{
					t.j = v;
					triplet1.push_back(t);
				}
			}
		}
	}
#pragma omp barrier
	setFromTriplets_lhs(A1, triplet1);
	gettimeofday(&time2,NULL);
	run_time= ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for building Gp: " << run_time / 1000 << "ms\n";
	cout << "Edges in Gps: " << nonZeros(A1) <<endl;

	parallel_core_decomposition(A1);
	gettimeofday(&time_over,NULL);
	run_time= ((time_over.tv_usec-time_start.tv_usec)+(time_over.tv_sec-time_start.tv_sec)*1000000); //us
	cout << "running time of HomBCore:" << run_time / 1000 << "ms\n\n";
	HomBCore_runtime = run_time;
}

void parallel_core_decomposition(matrix Adjacent)
{
	omp_lock_t lock;
	omp_init_lock(&lock);
	int* degree = new int[g.n + 1];
	vector<int> V;
	vector<neighbor> neighbors;
	initializeDegree(Adjacent, degree, V, neighbors);
	bool flg = 1;
	while (flg)
	{
		flg = 0;
#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < V.size(); i++)
		{
			int new_degree = 0;
			vector<int> cur_neighbors = neighbors[V[i]];
			if (cur_neighbors.size() == 0)
			{
				new_degree = 0;
			}
			else
			{
				int h_index = 0;
				int max_h = cur_neighbors.size();
				int* sum = new int[max_h + 1];
				int tot = 0;
				for (int k = 1; k <= max_h; k++) sum[k] = 0;

				for (int k = 0; k < max_h; k++)
				{
					if (degree[cur_neighbors[k]] >= max_h) sum[max_h]++;
					else sum[degree[cur_neighbors[k]]]++;
				}

				for (int k = max_h; k > 0; k--)
				{
					tot += sum[k];
					if (tot >= k)
					{
						new_degree = k;
						break;
					}
				}
			}

			if (new_degree != degree[V[i]])
			{
				degree[V[i]] = new_degree;
				omp_set_lock(&lock);
				flg = 1;
				omp_unset_lock(&lock);
			}
		}
#pragma omp barrier
	}
	omp_destroy_lock(&lock);

#pragma omp parallel for num_threads(NUM_THREADS)
	for(int i=0;i<g.n;i++) core[i]=degree[i];
#pragma omp barrier
}

void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor>& neighbors)
{
	vector<int> empty_neighbors;
	for (int r = 0; r < g.n; r++) neighbors.push_back(empty_neighbors);
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int r = 0; r < g.n; r++)
	{
		if (Adjacent.lhs_index[r] != Adjacent.lhs_index[r + 1])
		{
			int bg = Adjacent.lhs_index[r];
			int ed = Adjacent.lhs_index[r + 1] - 1;
			degree[r] = ed - bg;
			for (int element = bg; element <= ed; element++)
				if (Adjacent.lhs_elements[element].j != r)
				{
					neighbors[r].push_back(Adjacent.lhs_elements[element].j);
				}
#pragma omp critical
			{
				V.push_back(r);
			}
		}
		else degree[r] = 0;
	}
#pragma omp barrier
}

/****************************************************************************************************************************/
/************************************************  Matrix Related Functions  ************************************************/
/****************************************************************************************************************************/

bool cmp1(Triplet a, Triplet b)
{
	if (a.i == b.i) return a.j < b.j;
	else return a.i < b.i;
}

void setFromTriplets_lhs(matrix& A, triplet t)
{
	if (t.size() == 0)
	{
		for (int r = 0; r < g.n; r++) A.lhs_index.push_back(0);
		return;
	}

	sort(t.begin(), t.end(), cmp1);
	Index idx = 0;	
	int row = t[0].i;
	for (int r = 0; r <= row; r++) A.lhs_index.push_back(idx);
	for (int element = 0; element < t.size(); element++)
	{
		A.lhs_elements.push_back({t[element].j, t[element].value});
		if (t[element].i > row)
		{
			idx = element;
			for (int index = row + 1; index <= t[element].i; index++) A.lhs_index.push_back(idx);
			row = t[element].i;
		}
	}
	for (int r = row + 1; r <= A.row; r++) A.lhs_index.push_back(t.size());
}

int nonZeros(matrix A)
{
	return A.lhs_elements.size();
}
