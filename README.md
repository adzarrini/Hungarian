# Hungarian
Serial and parallel implementation of the Hungarian algorithm.  

To build the project, simply run `make` within the project directory. Call all `c++` file commands from the project directory.  

To run serial implementation, run:  
`./hungarian_serial ./matrix/<matrix-file-name> <max/min> <0/1>`  
example:  
`./hungarian_serial ./matrix/test.txt max 1`  

To check an assignment, use `check_hungarian.R` with the following command:  
`Rscript check_hungarian.R ./matrix/<matrix-file-name> <max/min> <0/1>`  
example:  
`Rscript check_hungarian.R matrix/test.txt min 0`

To create a bipartite cost matrix in the `./matrix` directory, run:  
`./creatrix <n> <max-rand-val> <random seed> <matrix-name (exclude ./matrix in file name)>`  
example:  
`./creatrix 3 9 9999 test.txt`
