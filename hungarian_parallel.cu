//  starter code taken from https://www.topcoder.com/community/competitive-programming/tutorials/assignment-problem-and-hungarian-algorithm/
//
//  edited by Joseph Greshik and Allee Zarrini

#include<iostream>
#include<stdio.h>
#include<string>
#include<fstream>
#include<cstring>

#define INF 100000000 //just infinity
#define THREADS_PER_BLOCK 16

int *cost; //cost matrix
int *dcost;

int n, max_match;//n workers and n jobs
int bytes; //bytes dependent on n

int *lx, *ly; //labels of X and Y parts
int *dlx, *dly;

int *xy; //xy[x] - vertex that is matched with x,
int *dxy;
int *yx; //yx[y] - vertex that is matched with y
int *dyx;

bool *S, *T; //sets S and T in algorithm
bool *dS, *dT;

int *slack; //as in the algorithm description
int *dslack;
int *slackx; //slackx[y] such a vertex, that
int *dslackx;
// l(slackx[y]) + l(y) - w(slackx[y],y) = slack[y]

int *prev; //array for memorizing alternating paths
int *dprev;

bool verbose=false;
bool maximum=false;

__global__ void init_labels(int n, int* lx, int* ly, int *cost) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) return;

	lx[x] = 0;
	ly[x] = 0;
	for (int y = 0; y < n; y++) 
		lx[x] = max(lx[x], cost[x*n+y]);
}


//void init_labels_s()
//{
//    memset(lx, 0, sizeof(int)*n);
//    memset(ly, 0, sizeof(int)*n);
//    for (int x = 0; x < n; x++)
//        for (int y = 0; y < n; y++)
//            lx[x] = std::max(lx[x], cost[x][y]);
//}

//__device__ void add_to_tree(int x, 
//							int prevx, 
//							int n, 
//							bool* S, 
//							int* prev, 
//							int* lx, 
//							int* ly, 
//							int* cost, 
//							int* slack, 
//							int* slackx)
//    //x - current vertex,prevx - vertex from X before x in the alternating path,
//    //so we add edges (prevx, xy[x]), (xy[x], x)
//{
//    int y = blockIdx.x * blockDim.x + threadIdx.x;
//	if (y >= n) return;
//
//    //update slacks, because we add new vertex to S
//    
//    if (lx[x] + ly[y] - cost[x*n+y] < slack[y])
//    {
//        slack[y] = lx[x] + ly[y] - cost[x*n+y];
//        slackx[y] = x;
//    }
//}

void add_to_tree(int x, int prevx)
    //x - current vertex,prevx - vertex from X before x in the alternating path,
    //so we add edges (prevx, xy[x]), (xy[x], x)
{
    S[x] = true; //add x to S
    prev[x] = prevx; //we need this when augmenting
    for (int y = 0; y < n; y++) //update slacks, because we add new vertex to S
        if (lx[x] + ly[y] - cost[x*n+y] < slack[y])
        {
            slack[y] = lx[x] + ly[y] - cost[x*n+y];
            slackx[y] = x;
        }
}

//__global__ void update_labels(int n, 
//							  bool* T, 
//							  bool* S, 
//							  int* lx, 
//							  int* ly, 
//							  int* slack)
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	if (x >= n) return;
//
//	__shared__ int delta;
//	delta = 100000000;
//
//	//calculate delta using slack
//    if (!T[x])
//        atomicMin(&delta, slack[x]);
//    
//	//update X labels
//    __syncthreads();
//
//	if (S[x]) lx[x] -= delta;
//	if (T[x]) ly[x] += delta;
//	else
//    	slack[x] -= delta;
//	
//}

void update_labels()
{
    int x, delta = INF; //init delta as infinity
    for (x = 0; x < n; x++) //calculate delta using slack
        if (!T[x])
            delta = std::min(delta, slack[x]);
    for (x = 0; x < n; x++) //update X labels
    {
		if (S[x]) lx[x] -= delta;
        if (T[x]) ly[x] += delta;
        else
            slack[x] -= delta;
	}
}

void augment(int n,
						int* cost,
						int* lx,
						int* ly,
						int* xy,
						int* yx,
						bool* S,
						bool* T,
						int* slack,
						int* slackx,
						int* prev) //main function of the algorithm
{
    if (max_match == n) return; //check wether matching is already perfect
    int x, y, root; //just counters and root vertex
    int *q, wr = 0, rd = 0; //q - queue for bfs, wr,rd - write and read
    q = new int[n];
    //pos in queue
    memset(S, false, sizeof(bool)*n); //init set S
    memset(T, false, sizeof(bool)*n); //init set T
    memset(prev, -1, sizeof(int)*n); //init set prev - for the alternating tree
    for (x = 0; x < n; x++) //finding root of the tree
        if (xy[x] == -1)
        {
            q[wr++] = root = x;
            prev[x] = -2;
            S[x] = true;
            break;
        }
    for (y = 0; y < n; y++) //initializing slack array
    {
        slack[y] = lx[root] + ly[y] - cost[root*n+y];
        slackx[y] = root;
    }
    //second part of augment() function
    while (true) //main cycle
    {
        while (rd < wr) //building tree with bfs cycle
        {
            x = q[rd++]; //current vertex from X part
            for (y = 0; y < n; y++) //iterate through all edges in equality graph
                if (cost[x*n+y] == lx[x] + ly[y] && !T[y])
                {
                    if (yx[y] == -1) break; //an exposed vertex in Y found, so
                    //augmenting path exists!
                    T[y] = true; //else just add y to T,
                    q[wr++] = yx[y]; //add vertex yx[y], which is matched
                    //with y, to the queue
        			
					//S[yx[y]] = true; //add x to S
    				//prev[yx[y]] = x; //we need this when augmenting
					//
					//cudaMemcpy(S, dS, sizeof(bool)*n, cudaMemcpyHostToDevice);
					//cudaMemcpy(prev, dprev, bytes, cudaMemcpyHostToDevice);
            		//cudaMemcpy(lx, dlx, bytes, cudaMemcpyHostToDevice);
					//cudaMemcpy(ly, dly, bytes, cudaMemcpyHostToDevice);
					//cudaMemcpy(cost, dcost, bytes*n, cudaMemcpyHostToDevice);
					//cudaMemcpy(slack, dslack, bytes, cudaMemcpyHostToDevice);
					//cudaMemcpy(slackx, dslackx, bytes, cudaMemcpyHostToDevice);
					//
					add_to_tree(yx[y], x);
					//add_to_tree<<<bytes/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(yx[y], x, n, dS, dprev, dlx, dly, dcost, dslack, dslackx); //add edges (x,y) and (y,yx[y]) to the tree
    				//cudaMemcpy(dS, S, sizeof(bool)*n, cudaMemcpyDeviceToHost);
					//cudaMemcpy(dprev, prev, bytes, cudaMemcpyDeviceToHost);
            		//cudaMemcpy(dlx, lx, bytes, cudaMemcpyDeviceToHost);
					//cudaMemcpy(dly, ly, bytes, cudaMemcpyDeviceToHost);
					//cudaMemcpy(dcost, cost, bytes*n, cudaMemcpyDeviceToHost);
					//cudaMemcpy(dslack, slack, bytes, cudaMemcpyDeviceToHost);
					//cudaMemcpy(dslackx, slackx, bytes, cudaMemcpyDeviceToHost);
					
				}
            if (y < n) break; //augmenting path found!
        }
        if (y < n) break; //augmenting path found!
		
		//cudaMemcpy(T, dT, sizeof(bool)*n, cudaMemcpyHostToDevice);
		//cudaMemcpy(S, dS, sizeof(bool)*n, cudaMemcpyHostToDevice);
        //cudaMemcpy(lx, dlx, bytes, cudaMemcpyHostToDevice);
		//cudaMemcpy(ly, dly, bytes, cudaMemcpyHostToDevice);
		//cudaMemcpy(slack, dslack, bytes, cudaMemcpyHostToDevice);

	  	update_labels();
		//update_labels<<<bytes/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(n,  dT, dS, dlx, dly, dslack); //augmenting path not found, so improve labeling
        //
        //cudaMemcpy(dlx, lx, bytes, cudaMemcpyDeviceToHost);
		//cudaMemcpy(dly, ly, bytes, cudaMemcpyDeviceToHost);
		//cudaMemcpy(dslack, slack, bytes, cudaMemcpyDeviceToHost);

		wr = rd = 0;
        for (y = 0; y < n; y++)
            //in this cycle we add edges that were added to the equality graph as a
            //result of improving the labeling, we add edge (slackx[y], y) to the tree if
            //and only if !T[y] && slack[y] == 0, also with this edge we add another one
            //(y, yx[y]) or augment the matching, if y was exposed
            if (!T[y] && slack[y] == 0)
            {
                if (yx[y] == -1) //exposed vertex in Y found - augmenting path exists!
                {
                    x = slackx[y];
                    break;
                }
                else
                {
                    T[y] = true; //else just add y to T,
                    if (!S[yx[y]])
                    {
                        q[wr++] = yx[y]; //add vertex yx[y], which is matched with
                        //y, to the queue
                    
					//S[yx[y]] = true; //add x to S
    				//prev[yx[y]] = slackx[y]; //we need this when augmenting
					//
					//cudaMemcpy(S, dS, sizeof(bool)*n, cudaMemcpyHostToDevice);
					//cudaMemcpy(prev, dprev, bytes, cudaMemcpyHostToDevice);
            		//cudaMemcpy(lx, dlx, bytes, cudaMemcpyHostToDevice);
					//cudaMemcpy(ly, dly, bytes, cudaMemcpyHostToDevice);
					//cudaMemcpy(cost, dcost, bytes*n, cudaMemcpyHostToDevice);
					//cudaMemcpy(slack, dslack, bytes, cudaMemcpyHostToDevice);
					//cudaMemcpy(slackx, dslackx, bytes, cudaMemcpyHostToDevice);

					add_to_tree(yx[y],slackx[y]);
                    //add_to_tree<<<bytes/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(yx[y], slackx[y], n, dS, dprev, dlx, dly, dcost, dslack, dslackx); //add edges (x,y) and (y,yx[y]) to the tree
                    //    //yx[y]) to the tree
                 	//cudaMemcpy(dS, S, sizeof(bool)*n, cudaMemcpyDeviceToHost);
                 	//cudaMemcpy(dprev, prev, bytes, cudaMemcpyDeviceToHost);
                 	//cudaMemcpy(dlx, lx, bytes, cudaMemcpyDeviceToHost);
                 	//cudaMemcpy(dly, ly, bytes, cudaMemcpyDeviceToHost);
                 	//cudaMemcpy(dcost, cost, bytes*n, cudaMemcpyDeviceToHost);
                 	//cudaMemcpy(dslack, slack, bytes, cudaMemcpyDeviceToHost);
                 	//cudaMemcpy(dslackx, slackx, bytes, cudaMemcpyDeviceToHost);

					}
                }
            }
        if (y < n) break; //augmenting path found!
    }
    if (y < n) //we found augmenting path!
    {
        max_match++; //increment matching
        //in this cycle we inverse edges along augmenting path
        for (int cx = x, cy = y, ty; cx != -2; cx = prev[cx], cy = ty)
        {
            ty = xy[cx];
            yx[cy] = cx;
            xy[cx] = cy;
        }
        augment(); //recall function, go to step 1 of the algorithm
    }
}//end of augment() function

int hungarian()
{
	int ret = 0;
	max_match = 0;
	
	memset(xy, -1, bytes);
	memset(yx, -1, bytes);
	cudaMemset(dxy, -1, bytes);
	cudaMemset(dyx, -1, bytes);
	
	init_labels<<<bytes/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(n, dlx, dly, dcost);
	
	cudaMemcpy(dlx, lx, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(dly, ly, bytes, cudaMemcpyDeviceToHost);

	augment(); //steps 1-3
    for (int x = 0; x < n; x++) {//forming answer there
        if (maximum) ret += cost[x*n+xy[x]];
        else ret += -cost[x*n+xy[x]];
    }
    return ret;
}

//int hungarian()
//{
//    int ret = 0; //weight of the optimal matching
//    max_match = 0; //number of vertices in current matching
//    memset(xy, -1, sizeof(int)*n);
//    memset(yx, -1, sizeof(int)*n);
//    init_labels(); //step 0
//    augment(); //steps 1-3
//    for (int x = 0; x < n; x++) {//forming answer there
//        if (maximum) ret += cost[x][xy[x]];
//        else ret += -cost[x][xy[x]];
//    }
//    return ret;
//}

void output_assignment()
{
    std::cout<<std::endl;
    for (int x = 0; x < n; x++){ //forming answer there
        if (maximum) std::cout<<cost[x*n+xy[x]]<<"\t";
        else std::cout<<-cost[x*n+xy[x]]<<"\t";
    }
    std::cout<<std::endl<<std::endl;
    std::cout<<"Optimal assignment: "<<std::endl;
    for (int x = 0; x < n; x++){ 
        std::cout<<x+1<<" => "<<xy[x]+1;
        if(x!=n-1) std::cout<<", ";
    }
    std::cout<<std::endl;
}

void read_in_cost_matrix(char* filename)
{
    std::ifstream fin (filename);
    fin>>n;

	bytes = sizeof(int)*n;

    cost = new int[n*n];
	cudaMalloc(&dcost, bytes*n);

    lx = new int[n]; ly = new int[n];
    cudaMalloc(&dlx, bytes); cudaMalloc(&dly, bytes);

	xy = new int[n];
    cudaMalloc(&dxy, bytes);
	yx = new int[n];
	cudaMalloc(&dyx, bytes);

    S = new bool[n]; T = new bool[n];
	cudaMalloc(&dS, sizeof(bool)*n); cudaMalloc(&dT, sizeof(bool)*n);

    slack = new int[n];
	cudaMalloc(&dslack, bytes);
    slackx = new int[n];
	cudaMalloc(&dslackx, bytes);

	prev = new int[n];
	cudaMalloc(&dprev, bytes);

    if (verbose) std::cout<<n<<std::endl;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            fin>>cost[i*n+j];
            if (!maximum) cost[i*n+j]=-cost[i*n+j];
            if (verbose)
            {
                if(maximum)  std::cout<<cost[i*n+j]<<"\t";
                else  std::cout<<-cost[i*n+j]<<"\t";
            }
        }
        if (verbose) std::cout<<std::endl;
    }
    fin.close();

	cudaMemcpy(dcost, cost, bytes*n, cudaMemcpyHostToDevice);

    // std::cout<<"Finished reading input" << std::endl;
}

int main(int argc, char*argv[])
{
    if (argc != 4) {
        std::cerr << "Arguments must be presented as follows." << std::endl;
        std::cerr << "./hungarian_parallel ./matrix/<matrix-file-name> <max/min> <0/1>" << std::endl;
        exit(1);
    }
    //    static const int arr[] = {7,4,2,8,2,3,4,7,1}; 
    verbose=atoi(argv[3]);
    if (std::string(argv[2])=="max") maximum=true;

    clock_t start, end;

    start = clock();
    read_in_cost_matrix(argv[1]);
    end = clock();
    double time_io = double(end - start) / double(CLOCKS_PER_SEC);
    // std::cout << "File IO: " << time_io << "s" << std::endl;

    start = clock();
    int x=hungarian();    
    end = clock();
    double time_algo = double(end - start) / double(CLOCKS_PER_SEC);

    if (verbose) output_assignment();

    std::cout<<n<<"\t\t"<<x<<"\t\t"<<time_algo<<std::endl;
    // std::cout<<"Algorithm execution: " << time_algo<<"s"<<std::endl;
}
