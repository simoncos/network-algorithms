# -*- coding: utf-8 -*-
'''
For the programming project, you will be given one data set. The data set will provide you the information
about the underlying graph. Note that the underlying graph is
• undirected,
• contains a single component,
• has no multi-edge (or a simple graph).

Degree centrality, eigenvector centrality, Katz centrality, pagerank centrality. You don’t need to print out the
centrality values of all nodes, but rather, state top 10 nodes ID and their centrality values.
• Repeat the above question but compute the betweenness and closeness centrality. Again, print out the top 10
nodes ID and their centrality values.
• Compute the degree distribution of the graph, and display the CCDF of the degree distribution and state whether
the underlying graph is a scale-free graph or not. If yes, what is the power-law exponent α?
• Determine local clustering coefficient of each node, and print out the top-10 and bottom-10 nodes IDs and their
local clustering coefficient.
'''

import itertools
import math 
from collections import Counter
import pylab # only for ccdf plot
from scipy import stats # only for ccdf slope fit

def file_to_sparse_matrix(file):
    '''
    Create a sparse sparse_matrix from the txt using dict.
    '''
    sparse_matrix = dict()
    with open(file) as f:
        for line in f:
            node_1, node_2 = line.split()
            if not node_1 in sparse_matrix:
                sparse_matrix[node_1] = set([node_2])               
            if not node_2 in sparse_matrix:
                sparse_matrix[node_2] = set([node_1])
            else:
                sparse_matrix[node_1].add(node_2)
                sparse_matrix[node_2].add(node_1)    
    return sparse_matrix

def print_tops(centrality, top=10):
    tops = sorted(centrality, key=lambda x: x[1], reverse=True)
    for i in tops[:top]:
        print('node {0}: {1}'.format(i[0],i[1]))

def print_bottoms(centrality, bottom=10):
    bottoms = sorted(centrality, key=lambda x: x[1], reverse=False)
    for i in bottoms[:bottom]:
        print('node {0}: {1}'.format(i[0],i[1]))

def degree_centrality(sparse_matrix):
    '''
    Node degree centrality.
    '''
    degree_centrality = list()
    for key, value in sparse_matrix.items():
        degree_centrality.append((key, len(value)))
    return degree_centrality

def degree_distribution(sparse_matrix):
    '''
    Node degree distribution CCDF
    '''
    node_degrees = degree_centrality(sparse_matrix)
    degrees = [i[1] for i in node_degrees]
    degree_distribution = Counter(degrees)

    # ccdf
    ccdf = {}
    for k in degree_distribution.keys():
        ccdf[k] = 1 - sum([degree_distribution[i] for i in degree_distribution.keys() if i<k]) / len(sparse_matrix)

    pylab.figure('ccdf')
    pylab.plot([k for k in ccdf.keys()], [p for p in ccdf.values()]) # plot ccdf
    pylab.xlabel('node degree')
    pylab.ylabel('ccdf value')
    pylab.title('ccdf of degree distribtion')
    pylab.show()
 
    # fit the exponent of powerlaw with log-log ccdf, ln p(k) = −α ln k + c 
    degree_distribution_prob = {}
    for k in degree_distribution.keys(): # change node count to prob
        degree_distribution_prob[k] = degree_distribution[k] / len(sparse_matrix)
    X = [math.log(k) for k in degree_distribution_prob.keys()]
    Y = [math.log(p) for p in degree_distribution_prob.values()]
    exponent = -(stats.linregress(X, Y).slope)
    if exponent > 1:
        print('\nThis is scale-free graph with powerlaw exponent', exponent)

def print_header(func):
    def call_wrapper(*args, **kwargs):
        if kwargs.get("no_deco") is True:
            return func(*args, **kwargs)
        print('\ntop nodes with highest {}:'.format(func.__name__))
        return func(*args, **kwargs)
    return call_wrapper

def vector_difference(vector_1, vector_2):  
    return sum(abs(vector_1[n]-vector_2[n]) for n in vector_1.keys()) 

@print_header
def eigenvector_centrality(sparse_matrix, error=1.0e-8, max_iter=100):
    '''
    Node eigenvector centrality.
    '''
    # initialization
    sparse_vector = sparse_matrix.fromkeys(sparse_matrix, 1/len(sparse_matrix))

    # while vector_difference(sparse_vector, sparse_vector_old) > len(sparse_vector) * error and max_iter < 100:
    for i in range(max_iter):
        sparse_vector_old = sparse_vector
        sparse_vector = sparse_matrix.fromkeys(sparse_matrix, 0)
        for node_1, node_2s in sparse_matrix.items():
            for node_2 in node_2s:
                sparse_vector[node_1] += sparse_vector_old[node_2] # entry of simple matrix is 0/1; undirected graph
        norm = sum(sparse_vector.values()) # norm by the sum of the centralities
        sparse_vector = {key: value/norm for key, value in sparse_vector.items()}
        if vector_difference(sparse_vector, sparse_vector_old) <= len(sparse_vector) * error:
            break

    eigenvector_centrality = list()
    if vector_difference(sparse_vector, sparse_vector_old) > len(sparse_vector) * error:
        print(sparse_vector)
        raise Exception('Convergence failed.')
    else:
        for key, value in sparse_vector.items():
            eigenvector_centrality.append((key, value))
    return eigenvector_centrality 

@print_header
def katz_centrality(sparse_matrix, error=1.0e-8, max_iter=100, alpha=0.8, beta=0.2):
    '''
    Node katz centrality.
    '''
    # initialization
    sparse_vector = sparse_matrix.fromkeys(sparse_matrix, 1/len(sparse_matrix))

    for i in range(max_iter):
        sparse_vector_old = sparse_vector
        sparse_vector = sparse_matrix.fromkeys(sparse_matrix, 0)
        for node_1, node_2s in sparse_matrix.items():
            for node_2 in node_2s:
                sparse_vector[node_1] += (alpha * sparse_vector_old[node_2] + beta) # entry of simple matrix is 0/1
        norm = sum(sparse_vector.values()) # norm by the sum of the centralities
        sparse_vector = dict({key: value/norm for key, value in sparse_vector.items()})
        if vector_difference(sparse_vector, sparse_vector_old) <= len(sparse_vector) * error:
            break

    katz_centrality = list()
    if vector_difference(sparse_vector, sparse_vector_old) > len(sparse_vector) * error:
        print(sparse_vector)
        raise Exception('Convergence failed.')
    else:
        for key, value in sparse_vector.items():
            katz_centrality.append((key, value))
    return katz_centrality 

@print_header
def pagerank_centrality(sparse_matrix, error=1.0e-8, max_iter=100, alpha=0.8, beta=0.2):
    '''
    Node pagerank centrality.
    '''
    # initialization
    sparse_vector = sparse_matrix.fromkeys(sparse_matrix, 1/len(sparse_matrix))

    for i in range(max_iter):
        sparse_vector_old = sparse_vector
        sparse_vector = sparse_matrix.fromkeys(sparse_matrix, 0)
        for node_1, node_2s in sparse_matrix.items():
            for node_2 in node_2s:
                # entry of simple matrix is 0/1
                sparse_vector[node_1] += (alpha * sparse_vector_old[node_2] / len(sparse_matrix[node_2]) + beta) 
        norm = sum(sparse_vector.values()) # norm by the sum of the centralities
        sparse_vector = dict({key: value/norm for key, value in sparse_vector.items()})
        if vector_difference(sparse_vector, sparse_vector_old) <= len(sparse_vector) * error:
            break

    pagerank_centrality = list()
    if vector_difference(sparse_vector, sparse_vector_old) > len(sparse_vector) * error:
        print(sparse_vector)
        raise Exception('Convergence failed.')
    else:
        for key, value in sparse_vector.items():
            pagerank_centrality.append((key, value))
    return pagerank_centrality 

def shortest_path_distance(sparse_matrix):
    '''
    Calculate shortest path distance and number of shortest paths 
    for all pairs of nodes; used in closeness_centrality()
    '''    

    # initialize the shortest path dict
    shortest_path_dict = {}
    for node_1 in sparse_matrix.keys():
        shortest_path_dict[(node_1, node_1)] = [0, 0] # shortest path length, number of shortest paths
    node_pair = list(itertools.permutations(sparse_matrix.keys(), 2)) # len = n(n-1)
    for np in node_pair:
        shortest_path_dict[np] = [-1, 0]

    for node_1 in sparse_matrix.keys():
        d = 0
        neighbors = sparse_matrix[node_1] # set
        for i in range(d+1):
            for un in unknown_neighbors:
                shortest_path_dict[(node_1, un)][1] = d + 1
                unknown_neighbors = unknown_neighbors.union(sparse_matrix(un)) # new object union
                unknown_neighbors.remove(un) # inner remove    
                
def shortest_path(sparse_matrix):
    '''
    Calculate shortest paths for all pairs of nodes; 
    used in betweenness_centrality()
    '''    


@print_header
def closeness_centrality(sparse_matrix):
    '''
    Node closeness centrality.
    '''
    closeness_centrality = list()
    
    return closeness_centrality 

@print_header
def betweenness_centrality(sparse_matrix):
    '''
    Node betweenness centrality.
    '''
    betweenness_centrality = list()
    
    return betweenness_centrality 

def local_clustering_coefficient(sparse_matrix):
    '''
    Node clustering coefficient.
    '''
    local_clusteing_coefficient = list()

    for node, node_2s in sparse_matrix.items():
        if len(node_2s) <= 1:
            local_clusteing_coefficient.append((node, 0.0))
        else:
            connected_pair_count = 0
            pair_count = len(list(itertools.combinations(node_2s, 2)))
            for p in itertools.combinations(node_2s, 2): # after iteration, the itertools.combinations will reduce
                if p[1] in sparse_matrix[p[0]]:
                    connected_pair_count += 1
            local_clusteing_coefficient.append((node, connected_pair_count / pair_count))

    return local_clusteing_coefficient

def main():
    sparse_matrix = file_to_sparse_matrix('graph.txt')

    print('top nodes with highest degree_centrality:')
    print_tops(degree_centrality(sparse_matrix))
    degree_distribution(sparse_matrix)    

    print_tops(eigenvector_centrality(sparse_matrix))
    print_tops(katz_centrality(sparse_matrix))
    print_tops(pagerank_centrality(sparse_matrix))
 
    print('\ntop nodes with highest local_clustering_coefficent:')
    print_tops(local_clustering_coefficient(sparse_matrix))
    print('\nbottom nodes with lowest local_clustering_coefficent:')
    print_bottoms(local_clustering_coefficient(sparse_matrix))
    

if __name__ == '__main__':
    main()