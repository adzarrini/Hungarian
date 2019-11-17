#!/usr/bin/env Rscript
## this is a script to check hungarian algorithm output
## author: Joseph Greshik

## clue library contains solve_LSAP function for 
##  computing hungarian soln given nxn adjacency
##  matrix for bipartite cost graph
library(clue)

## read in command line args
##  first arg should be matrix file name
##  second arg should be max or min
##  third arg should be integer 0 or 1
##      1 for printing out matrix
##      0 for no print of matrix
args = commandArgs(trailingOnly=TRUE)

## read in file to populate our adjacency matrix
filepath=args[1]
processFile = function(filepath) {
    con = file(filepath, "r")
    n=readLines(con, n=1)
    n=strtoi(n)
    indices<-c(n*n)
    for (i in 1:(n*n)) {
        line = readLines(con, n = 1)
        indices[i]<-strtoi(line[1])
    }
    close(con)
    list(n=n,indices=indices)
}
file_mat<-processFile(filepath)
adj_mat<-matrix(
             data=file_mat$indices,
             nrow=file_mat$n,
             ncol=file_mat$n,
             byrow=TRUE)

if (args[3]==1) print(adj_mat)

if (args[2]=='max') maximum<-TRUE
if (args[2]=='min') maximum<-FALSE

hungarian<-solve_LSAP(x=adj_mat,maximum=maximum)
print(hungarian)
print(sum(adj_mat[cbind(seq_along(hungarian), hungarian)]))
print(adj_mat[cbind(seq_along(hungarian), hungarian)])
