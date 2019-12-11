#include<iostream>
#include<fstream>

using namespace std;

bool verbose, maximum;

int n;
int *C; 		// C matrix (nxn)
int *Ar, *Ac; 	// assignment arrays
bool *Vr, *Vc;	// cover arrays
int *Dr, *Dc;	// dual variable
int *slack;		// slack array
int *Pr, *Pc;	// predecessor arrays
int *Sr, *Sc;	// successor arrays

__global__ void initKernalReduc(int n, int *C, int *Dr, int *Dc) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i >= n) return;
	
	int tmin = 100000000; // very large number
	for(int j = 0; j < n; j++) {
		tmin = min(tmin, C[i*n+j]);
	}
	Dr[i] = tmin;

	__syncthreads();
	
	tmin = 100000000;
	for (int j = 0; j < n; j+=) {
		tmin = min(tmin, C[j*n+i] - Dr[j]);
	}
	Dc[i] = tmin;
}

int hungarian() 
{
	int max_match = 0;

	int bytes = sizeof(int)*n;
	
	int *dC;
	cudaMalloc(&dC, bytes*n);
	cudaMemcpy(dC, C, bytes*n, cudaMemcpyHostToDevice); 
	
	printCost();
	
	
	
	return max_match;
}

void read_in_C_matrix(char* filename)
{
    ifstream fin (filename);
    fin>>n;

    C = new int[n*n];
        
	if (verbose) cout<<n<<endl;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            fin>>C[i*n+j];
            if (!maximum) C[i*n+j]=-C[i*n+j];
            if (verbose)
            {
                if(maximum)  cout<<C[i*n+j]<<"\t";
                else  cout<<-C[i*n+j]<<"\t";
            }
        }
        if (verbose) cout<<endl;
    }
    fin.close();
    //  cout<<"Finished reading input" << endl;
}

void output_assignment()
{
    cout<<endl;
    for (int i = 0; i < n; i++){ //forming answer there
        if (maximum) cout<<C[i*n+Ar[i]]<<"\t";
        else cout<<-C[i*n+Ar[i]]<<"\t";
    }
    cout<<endl<<endl;
    cout<<"Optimal assignment: "<<endl;
    for (int i = 0; i < n; i++){ 
        cout<<i+1<<" => "<<Ar[i]+1;
        if(i!=n-1) cout<<", ";
    }
    cout<<endl;
}

void printArray(int* arr) {
	for (int i = 0; i < n; i++) {
		cout << arr[i] << " ";
	}
	cout << endl;
}

void printCost()
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << C[i*n+j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main(int argc, char*argv[])
{
    if (argc != 4) {
        cerr << "Arguments must be presented as follows." << endl;
        cerr << "./hungarian_parallel ./matrix/<matrix-file-name> <max/min> <0/1>" << endl;
        exit(1);
    }
    //    static const int arr[] = {7,4,2,8,2,3,4,7,1}; 
    verbose=atoi(argv[3]);
    if (string(argv[2])=="max") maximum=true;

    clock_t start, end;

    start = clock();
    read_in_C_matrix(argv[1]);
    end = clock();
    //double time_io = double(end - start) / double(CLOCKS_PER_SEC);
    // cout << "File IO: " << time_io << "s" << endl;

    start = clock();
    int x=hungarian();    
    end = clock();
    double time_algo = double(end - start) / double(CLOCKS_PER_SEC);

    if (verbose) output_assignment();

    cout<<n<<"\t\t"<<x<<"\t\t"<<time_algo<<endl;
    // cout<<"Algorithm execution: " << time_algo<<"s"<<endl;
}

