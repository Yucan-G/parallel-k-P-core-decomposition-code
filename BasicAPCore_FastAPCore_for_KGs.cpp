/*********** Code for BasicAPCore&FastAPCore (Optimizations for KGs) ***********/

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
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
int NUM_THREADS;

/*path of input files*/
char* HIN_PATH = (char*)"Movielens/graph_Movielens.txt";
char* METAPATH_PATH = (char*)"Movielens/metapath.txt";

/*data structures*/
typedef struct {
	int v;
	int type;
}edge;
typedef vector<edge> vertice;

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
}Tuple;
typedef vector <Tuple> tuples;

typedef int Index;

typedef struct {
	int row, col;
	vector<Index> lhs_index, rhs_index;
	vector<int> lhs_elements, rhs_elements;
}matrix;

vector<matrix> A;
vector<tuples> edgeTuples;
vector<int> path;
vector<int> core;
int path_length;
double BasicAPCore_runtime, FastAPCore_runtime;

void read_env();
int load_HIN(char* filename);
int load_metapath(char* filename);
void BasicAPCore();
void FastAPCore();
void parallel_core_decomposition(matrix Adjacent);
void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor> &neighbors);

/*matrix related functions*/
void setFromTuples(matrix &A, tuples t);
matrix transpose1(matrix A); //from rhs to lhs
matrix transpose2(matrix A); //from lhs to lhs
matrix sparseMatrixProduct(matrix A, matrix B);
int nonZeros(matrix A);


int main(int argc, char* argv[])
{
	read_env();
	load_HIN(HIN_PATH);
	load_metapath(METAPATH_PATH); 
	BasicAPCore();
	FastAPCore();
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
		fprintf(stderr, "Error: could not open inputh file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading HIN from: %s\n", filename);
	fscanf(infp, "%d %d %d\n", &(g.n), &(g.m), &(g.t));
	vertice tmp_v; tuples tmp_t; matrix tmp_m;
	tmp_m.row = tmp_m.col = g.n;
	for (int i = 0; i < g.n; i++) { g.v.push_back(tmp_v); edgeTuples.push_back(tmp_t); core.push_back(0);}
	for (int i = 0; i < g.t; i++) A.push_back(tmp_m);
	int u, v, type;
	edge e;
	while (fscanf(infp, "%d %d %d\n", &u, &v, &type) != EOF) {
		e.v = v; e.type = type;
		g.v[u].push_back(e);
		e.v = u; e.type = -type;
		g.v[v].push_back(e);
		Tuple t = {u, v};
		edgeTuples[type - 1].push_back(t);
	}
	fclose(infp);
	for (int i = 0; i < g.t; i++)
	{
		setFromTuples(A[i], edgeTuples[i]);
	}
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
	fscanf(infp, "%d\n", &path_length);
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

void BasicAPCore()
{
	double run_time;
	timeval time_start;
	timeval time_over;
	timeval time1, time2;
	cout << "Ready to run BaiscAPCore..." << endl;
	gettimeofday(&time_start,NULL);
	gettimeofday(&time1,NULL);

	matrix B = { g.n, g.n };
	long long ini_type = path[0];
	if (ini_type > 0) B = A[ini_type - 1];
	else B = transpose1(A[abs(ini_type) - 1]);
	for (int l = 1; l < path_length; l++)
	{
		if (path[l] > 0) B = sparseMatrixProduct(B, A[path[l] - 1]);
		else B = sparseMatrixProduct(B, transpose1(A[abs(path[l]) - 1]));
	}

	gettimeofday(&time2,NULL);
	run_time = ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for building Gp: " << run_time / 1000 << "ms\n";
	parallel_core_decomposition(B);

	gettimeofday(&time_over,NULL);
	run_time= ((time_over.tv_usec-time_start.tv_usec)+(time_over.tv_sec-time_start.tv_sec)*1000000); //us
	cout << "running time of BasicAPCore:" << run_time / 1000 << "ms\n\n";
	BasicAPCore_runtime = run_time;
}

void FastAPCore()
{
	double run_time;
	timeval time_start;
	timeval time_over;
	timeval time1, time2;
	cout << "Ready to run FastAPCore..." << endl;
	gettimeofday(&time_start,NULL);
	gettimeofday(&time1,NULL);
	matrix B = { g.n, g.n };
	int target_num = 0;
	if (path[0] > 0) B = A[path[0] - 1];
	else B = transpose1(A[abs(path[0]) - 1]);
	double mean_density = 0;
	for (long long i = 1; i < path_length / 2; i++)
	{
		mean_density += double(nonZeros(B)) / g.n / g.n;
		if (path[i] > 0) B = sparseMatrixProduct(B, A[path[i] - 1]);
		else B = sparseMatrixProduct(B, transpose1(A[abs(path[i]) - 1]));
	}
	if(path_length>2) 
	{
		cout << "mean density=" << mean_density / (path_length / 2 - 1) << endl;
		double cur_density = double(nonZeros(B)) / g.n / g.n;
		cout << "intermidiate density=" << cur_density << endl;
		double decision = 1.55584046 + 0.0695962647 * (mean_density / cur_density) + 241.063437 * mean_density -508.697917 * cur_density;
				
		if (decision < 1)
		{
			cout << "    *The matrix is not appropriate for transpose" << endl;
			for (long long i = path_length / 2; i < path_length; i++)
			{
				if (path[i] > 0) B = sparseMatrixProduct(B, A[path[i] - 1]);
				else B = sparseMatrixProduct(B, transpose1(A[abs(path[i]) - 1]));
			}
		}
		else B = sparseMatrixProduct(B, transpose2(B));
	}
	else
	{
	    if(path[0] > 0) B = sparseMatrixProduct(B, transpose1(B));
	    else B = sparseMatrixProduct(B, A[abs(path[0]) - 1]);
	}

	gettimeofday(&time2,NULL);
	run_time = ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for building Gp: " << run_time / 1000 << "ms\n";
	cout<<"Edges in Gp: "<<nonZeros(B)<<endl;

	gettimeofday(&time1,NULL);
	parallel_core_decomposition(B);
	gettimeofday(&time2,NULL);
	run_time = ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for core decomposition: " << run_time / 1000 << "ms\n";

	gettimeofday(&time_over,NULL);
	run_time = ((time_over.tv_usec-time_start.tv_usec)+(time_over.tv_sec-time_start.tv_sec)*1000000); //us
	cout << "running time of FastAPcore:" << run_time / 1000 << "ms\n";
	FastAPCore_runtime = run_time;
	cout << "BasicAPCore running time / FastAPCore running time = " << BasicAPCore_runtime / FastAPCore_runtime << endl << endl;
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
				if (Adjacent.lhs_elements[element] != r)
				{
					neighbors[r].push_back(Adjacent.lhs_elements[element]);
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

bool cmp1(Tuple a, Tuple b)
{
	if (a.i == b.i) return a.j < b.j;
	else return a.i < b.i;
}

bool cmp2(Tuple a, Tuple b)
{
	if (a.j == b.j) return a.i < b.i;
	else return a.j < b.j;
}

void setFromTuples(matrix& A, tuples t)
{
	if(t.size()==0)
	{
		Index empty_idx;
		vector <int> empty_elements;
		empty_idx = 0;
		for (int r = 0; r <= g.n; r++) A.lhs_index.push_back(empty_idx);
		A.lhs_elements = A.rhs_elements = empty_elements;
		for (int c = 0; c <= g.n; c++) A.rhs_index.push_back(empty_idx);
		return;
	}

	sort(t.begin(), t.end(), cmp1);
	int row = t[0].i;
	Index idx = 0;
	for (int r = 0; r <= row; r++) A.lhs_index.push_back(idx);
	for (int element = 0; element < t.size(); element++)
	{
		A.lhs_elements.push_back(t[element].j);
		if (t[element].i > row)
		{
			idx = element;
			for (int index = row + 1; index <= t[element].i; index++) A.lhs_index.push_back(idx);
			row = t[element].i;
		}
	}
	for (int r = row + 1; r <= A.row; r++) A.lhs_index.push_back(t.size());

	sort(t.begin(), t.end(), cmp2);
	int col = t[0].j;
	idx = 0;
	for (int c = 0; c <= col; c++) A.rhs_index.push_back(idx);
	for (int element = 0; element < t.size(); element++)
	{
		A.rhs_elements.push_back(t[element].i);
		if (t[element].j > col)
		{
			idx = element;
			for (int index = col + 1; index <= t[element].j; index++) A.rhs_index.push_back(idx);
			col = t[element].j;
		}
	}
	for (int c = col + 1; c <= A.col; c++) A.rhs_index.push_back(t.size());
}

matrix transpose1(matrix A) //from rhs to lhs
{
	matrix B;
	B.col = A.row;
	B.row = A.col;
	B.lhs_index = A.rhs_index;
	B.lhs_elements = A.rhs_elements;
	return B;
}

matrix transpose2(matrix A) //from lhs to lhs
{
	matrix B;
	tuples t;
	B.col = A.row;
	B.row = A.col;
	for(int r = 0; r < A.row; r++)
	{
		if(A.lhs_index[r] != A.lhs_index[r + 1])
		{
			int bg = A.lhs_index[r];
			int ed = A.lhs_index[r + 1] - 1;
			for(int element = bg; element <= ed; element++)
			{
				t.push_back({A.lhs_elements[element], r});
			}
		}
	}
	if(t.size()==0)
	{
		Index empty_idx;
		vector <int> empty_elements;
		empty_idx = -1;
		for (int r = 0; r <= g.n; r++) B.lhs_index.push_back(empty_idx);
		B.lhs_elements = empty_elements;
		return B;
	}

	sort(t.begin(), t.end(), cmp1);
	int row = t[0].i;
	Index idx = 0;
	for (int r = 0; r <= row; r++) B.lhs_index.push_back(idx);
	for (int element = 0; element < t.size(); element++)
	{
		B.lhs_elements.push_back(t[element].j);
		if (t[element].i > row)
		{
			idx = element;
			for (int index = row + 1; index <= t[element].i; index++) B.lhs_index.push_back(idx);
			row = t[element].i;
		}
	}
	for (int r = row + 1; r <= A.row; r++) B.lhs_index.push_back(t.size());
	return B;
}

matrix sparseMatrixProduct(matrix A, matrix B) //only need lhs format
{
	matrix C;
	C.row = A.row;
	C.col = B.col;
	vector<int> empty_elements;
	vector<vector<int>> elements;
	int last_size = 0, cur_size;
	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		elements.push_back(empty_elements);
	}
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		if (A.lhs_index[cur_row] != A.lhs_index[cur_row + 1])
		{
			int row_bg = A.lhs_index[cur_row];
			int row_ed = A.lhs_index[cur_row + 1] - 1;
			vector<int> sum;
			for (int element = row_bg; element <= row_ed; element++)
			{
				int j = A.lhs_elements[element];
				int b_row_bg = B.lhs_index[j];
				int b_row_ed = B.lhs_index[j + 1] - 1;
				if (b_row_bg == b_row_ed + 1) continue;
				for (int b_element = b_row_bg; b_element <= b_row_ed; b_element++)
				{
					sum.push_back(B.lhs_elements[b_element]);
				}
			}
			if (sum.size() == 0) continue;
			sort(sum.begin(), sum.end());
			int last_item = sum[0];
			for(int item = 1; item < sum.size(); item++)
			{
				int cur_item = sum[item];
				if(cur_item != last_item)
				{
					elements[cur_row].push_back(last_item);
					last_item = cur_item;
				} 
			}
			elements[cur_row].push_back(last_item);
		}
	}
	#pragma omp barrier
	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		int size = elements[cur_row].size();
		if (size > 0)
		{
			C.lhs_elements.insert(C.lhs_elements.end(), elements[cur_row].begin(), elements[cur_row].end());
			C.lhs_index.push_back(last_size);
			cur_size = last_size + size;
			last_size = cur_size;
		}
		else C.lhs_index.push_back(last_size);
	}
	C.lhs_index.push_back(last_size);
	return C;
}

int nonZeros(matrix A)
{
	return A.lhs_elements.size();
}
