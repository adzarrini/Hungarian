#include <iostream>
#include <fstream>
#include <queue>

using namespace std;

void minimizeMat(int **mat, int **tmat, int n);
void minimizeRows(int **mat, int **tmat, int n);
void minCostMaxFlow();
void printMat(int **mat, int n);

int n;
int **org, **mat, **tmat;
int **capacity, **flow;
int *height, *excess;

int main(int argc, char *argv[])
{

	if (argc != 2) {
		cerr << "Command-line call in the following way. \n" << "./executable matrix/input.txt" << endl; 
	}

	ifstream fin (argv[1]);

	fin >> n;

	org = new int*[n]; mat = new int*[n]; tmat = new int*[n];
	capacity = new int*[n]; flow = new int*[n];

	for (int i = 0; i < n; i++) {
		org[i] = new int[n]; mat[i] = new int[n]; tmat[i] = new int[n];
		capacity[i] = new int[n]; flow[i] = new int[n];
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fin >> org[i][j];
			mat[i][j] = org[i][j];
			tmat[j][i] = mat[i][j];
		}
	}
	
	fin.close();
	
	minimizeMat(mat, tmat, n);

	printMat(org, n);
	cout << endl;
	printMat(mat, n);

}

void minimizeMat(int **mat, int **tmat, int n)
{
	minimizeRows(mat, tmat, n);
	minimizeRows(tmat, mat, n);
}

void minimizeRows(int **mat, int **tmat, int n)
{
	int min;
	bool zero;
	for (int i = 0; i < n; i++)
	{
		int *row = mat[i];
		min = INT_MAX;
		zero = false;
		for (int j = 0; j < n; j++)
		{
			if (row[j] == 0) {
				zero = true;
				break;
			}
			min = (row[j] < min) ? row[j] : min;
		}

		if(zero) continue;
		for (int j = 0; j < n; j++)
		{
			row[j] -= min;
			tmat[j][i] = row[j];
		}
	}
}

void printMat(int **mat, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << mat[i][j] << "\t";
		}
		cout << endl;
	}
}


// Push-relabel


// void push(int u, int v)
// {
// 	int d = min(excess[u], capacity[u][v] - flow[u][v]);
// 	flow[u][v] += d;
// 	flow[v][u] -= d;
// 	excess[u] -= d;
// 	excess[v] += d;
// }

// void relabel(int u)
// {
// 	int d = inf;
// 	for (int i = 0; i < n; i++)
// 	{
// 		if (capacity[u][i] - flow[u][i] > 0)
// 			d = min(d, height[i]);
// 	}
// 	if (d < inf)
// 		height[u] = d + 1;
// }

// vector<int> find_max_height_vertices(int s, int t)
// {
// 	vector<int> max_height;
// 	for (int i = 0; i < n; i++)
// 	{
// 		if (i != s && i != t && excess[i] > 0)
// 		{
// 			if (!max_height.empty() && height[i] > height[max_height[0]])
// 				max_height.clear();
// 			if (max_height.empty() || height[i] == height[max_height[0]])
// 				max_height.push_back(i);
// 		}
// 	}
// 	return max_height;
// }

// int max_flow(int s, int t)
// {
// 	height.assign(n, 0);
// 	height[s] = n;
// 	flow.assign(n, vector<int>(n, 0));
// 	excess.assign(n, 0);
// 	excess[s] = inf;
// 	for (int i = 0; i < n; i++)
// 	{
// 		if (i != s)
// 			push(s, i);
// 	}

// 	vector<int> current;
// 	while (!(current = find_max_height_vertices(s, t)).empty())
// 	{
// 		for (int i : current)
// 		{
// 			bool pushed = false;
// 			for (int j = 0; j < n && excess[i]; j++)
// 			{
// 				if (capacity[i][j] - flow[i][j] > 0 && height[i] == height[j] + 1)
// 				{
// 					push(i, j);
// 					pushed = true;
// 				}
// 			}
// 			if (!pushed)
// 			{
// 				relabel(i);
// 				break;
// 			}
// 		}
// 	}

// 	int max_flow = 0;
// 	for (int i = 0; i < n; i++)
// 		max_flow += flow[0][i];
// 	return max_flow;
// }