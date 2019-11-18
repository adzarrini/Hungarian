#include<iostream>
#include<fstream>
#include<string>
#include<time.h>

using namespace std;

int main(int argc, char* argv[]){
    if (argc != 5) {
        cerr << "Arguments must be presented as follows." << endl;
        cerr << "./executable number(int) max(int) seed(int) filename.txt(string)" << endl;
        exit(1);
    }

    int n = atoi(argv[1]);
    int max = atoi(argv[2]);
    srand(atoi(argv[3]));
    ofstream fout("./matrix/"+string(argv[4]));

    fout<<n<<"\n";
    for(int k=0;k<n*n;++k){
        int i = (int) (((double) rand() / (RAND_MAX)) * max) + 1;
        fout<<i<<"\n";
    }
    fout.close();
    return 0;
}
