import networkx as nx
import numpy as np
import math

A = np.matrix('0 1 0 0 1 0; 1 0 1 0 1 0; 0 1 0 1 0 0; 0 0 1 0 1 1; 1 1 0 1 0 0; 0 0 0 1 0 0')
print("A = \n", A)
W = np.copy(A[np.nonzero(A)]).flatten()
print("W = \n", W)
G = nx.DiGraph(A)#nx.convert_matrix.from_numpy_matrix(A)
print("G = \n", G.nodes(),G.edges())

laplacian = nx.linalg.laplacianmatrix.directed_laplacian_matrix(G)#Form the laplacian of our input graph

incidence = nx.linalg.graphmatrix.incidence_matrix(G,oriented=True)#Form the incidence matrix of our input graph


print("Laplacian = \n", np.shape(laplacian),laplacian)
print("Incidence = \n", incidence)
epsilon = 0.01#Epsilon used for generating k
k = round(24*math.log2(G.number_of_nodes()/(epsilon**2)))#K used for generating the Q matrix
m = G.number_of_edges()
choices = [1/k**0.5,-1/k**0.5]
Q = np.random.choice(choices, (k,m))#Generating Q, a matrix of shape (k,m) with values of +/- 1/(k^0.5) (page 10)
print("Q = \n",Q)

W_mat = np.diag(W)
print("Wdiag = \n", W_mat)
incidence_np = incidence.todense()#Convert our incidence matrix to a numpy matrix to use the least squares method
print("shapes = ",np.shape(W_mat),np.shape(incidence),np.shape(Q))
Y = np.matmul(Q,np.sqrt(W_mat))#Computing the first part of Y (page 10)
Y = np.matmul(Y,incidence_np.transpose())#Computing the second part of Y
print("Y = \n", np.shape(Y),Y)
Z = np.linalg.lstsq(laplacian,Y.transpose())#Use the least squares method to replace the STSolve
print("Z = \n",np.shape(Z[0]))
Z_t = Z[0].copy()#.transpose()
Re = np.zeros(G.number_of_edges())
counter = 0
for start,end in G.edges():
    temp = np.copy((Z_t[start] - Z_t[end])).flatten()
    Re[counter] = np.linalg.norm(temp,ord=2,axis=None,keepdims=False)
    counter += 1
print("Re = \n",Re)
Re_norm = Re*W
Re_norm = Re_norm/Re_norm.sum()#Our normalized probability distribution of our edges (the change each edge will get sampled in our new graph)
print("Re_norm = \n",Re_norm)
q = 10#The number of edges to sample from our input graph into our sparsified graph
Re_weight = W/(Re_norm*q)#Adjust the weights of our sampled edges (page 2)
print("Re_weight = \n",Re_weight)
choices = list(range(G.number_of_edges()))#Our choices of edges to sample, by their index in the edge list of our input graph
edgesNew = np.random.choice(choices, q, p = Re_norm)#Our new set of edges for our sparsified graph
H = nx.DiGraph()#Defining our new sparsified graph object
for _ in range(G.number_of_nodes()):#Add all of our nodes to our new graph
    H.add_node(_)
for edge in edgesNew:#Add all of the sampled edges with their adjusted weights to the new graph H
    start, end = list(G.edges())[edge]
    if not H.has_edge(start,end): H.add_edge(start,end, weight=Re_weight[edge])
    else: H[start][end]["weight"] += Re_weight[edge]#In the case of an edge being sampled twice, sum the weights.

print("G edges = \n",G.edges())
print("\n",nx.adjacency_matrix(G).todense())
print("H edges = \n",H.edges())
print("\n",nx.adjacency_matrix(H).todense())