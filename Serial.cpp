#include <iostream>
#include <fstream>

using namespace std;

int minimizeRow(int **mat, int **tmat, int n);

int main(int argc, char* argv[]) {

	if (argc != 2) {
		cerr << "Command-line call in the following way. \n" << "./$(FILE) input.txt" << endl; 
	}

	ifstream fin (argv[1]);

	int n;
	fin >> n;

	int** mat = new int*[n];
	int** tmat = new int*[n];

	for (int i = 0; i < n; i++) {
		mat[i] = new int[n];
		tmat[i] = new int[n];
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fin >> mat[i][j];
			mat[j][i] = mat[i][j];
		}
	}

}

int minimizeRow(int *row, int n, int **tmat, int r) {
	int min = INT_MAX;
	int t;
	for (int i = 0; i < n; i++) {
		t = row[i];
		min = (t < min) ? t : min;
	}

	for (int i = 0; i < n; i++) {
		row[i] -= t;
		tmat[i][r] = row[i];
	}
        return 0;
}

int printMat(int **mat, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << mat[i][j] << " ";
		}
		cout << endl;
	}
        return 0;
}
