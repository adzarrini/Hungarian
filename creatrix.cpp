#include<fstream>
#include<string>
#include<time.h>
using namespace std;
int main(int argc, char* argv[]){
    srand(time(NULL)); 
    int n = atoi(argv[1]);
    int max = atoi(argv[2]);
    ofstream out("./matrix/"+string(argv[3]));
    out<<n<<"\n";
    for(int k=0;k<n*n;++k){
        int i = rand() % max + 1;
        out<<i<<"\n";
    }
    out.close();
    return 0;
}
