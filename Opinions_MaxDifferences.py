import numpy as np
import pandas as pd
import random
import scipy.stats
from scipy.stats import beta, binom, expon, multivariate_normal
import seaborn as sns
import math
import matplotlib.pyplot as plt
import networkx as nx
from networkx import *


# dSBM with 2 communities 
def dSBM(n, kappa, theta_n):
    
    n1 = n2 = n//2 # number of nodes in each community 
    
    A = np.zeros((n, n)) # initialize adjacency matrix
    
    # Block matrix with connection probabilities
    p11 = (kappa[0][0]*theta_n)/n
    p12 = (kappa[0][1]*theta_n)/n
    p21 = (kappa[1][0]*theta_n)/n
    p22 = (kappa[1][1]*theta_n)/n
    
    # Create adjacency matrix
    A[:n1, :n1] = np.random.rand(n1, n1) < p11 # community 1 to community 1
    np.fill_diagonal(A[:n1, :n1], 0) # remove self-loops for community 1 nodes
    
    A[:n1, n1:] = np.random.rand(n1, n2) < p12 # community 1 to community 2
    
    A[n1:, :n1] = np.random.rand(n2, n1) < p21 # community 2 to community 1
    
    A[n1:, n1:] = np.random.rand(n2, n2) < p22 # community 2 to community 2
    np.fill_diagonal(A[n1:, n1:], 0) # remove self-loops for community 2 nodes
    
    return A

def B_weights(n):
    
    n1 = n2 = n//2 # number of nodes in each community 
    
    B = np.zeros((n, n))
    
    #B[:n1, :n1] = np.random.normal(5, 0.1, size=(n1, n1)) # community 1 to community 1
    #B[:n1, n1:] = np.random.normal(2.5, 0.1, size=(n1, n2)) # community 1 to community 2
    #B[n1:, :n1] = np.random.normal(5, 0.1, size=(n2, n1)) # community 2 to community 1
    #B[n1:, n1:] = np.random.normal(2.5, 0.1, size=(n2, n2)) # community 2 to community 2
    
    B[:n1, :n1] = 4  # community 1 to community 1
    B[:n1, n1:] = 1  # community 1 to community 2
    B[n1:, :n1] = 1  # community 2 to community 1
    B[n1:, n1:] = 4  # community 2 to community 2
    
    return B

def C_weights(n, c, d, A):
    
    n1 = n2 = n//2 # number of nodes in each community 
    
    C = np.zeros((n, n)) # initialize weight matrix
    D = np.sum(A, axis=0) # vector that contains the in-degree of each node
    B = B_weights(n)
    
    for i in range(n):
        for j in range(n):
            if D[i]>0:
                C[i, j] = (c*B[i, j]*A[j, i])/(np.sum(B[i, :]*A[:, i]))
                
    # the diagonal takes different values
    for i in range(n):
        C[i, i] = 1-c-d
        
    return C

# We modify the remaining functions so that they are for a single topic

# internal opinions
def internal_opinions(n, internal):
    
    n1 = n2 = n//2 # number of nodes in each community 
    
    Q = np.zeros(n) # initialize vector of internal opinions
    
    if internal == 'uniform':
        Q = 2*np.random.rand(n)-1 # Unif(-1, 1)
        
    else:
        Q[:n1] = np.random.rand(n)-1 # Unif(-1, 0) for nodes in community 1
        Q[n1:] = np.random.rand(n) # Unif(0, 1) for nodes in communtiy 2
    
    return Q

# media signals
def media_signals(n, T, media, Q):
    
    # n is the number of nodes (people), T is the number of iterations 
    # media is the type of media signals people receive
    
    # We will generate all the media at once (since they are independent) and then call them in the main function
    Z = np.zeros((n, T)) # initialize the media matrix
    
    if media == 'uniform':
        Z = -1 + 2*np.random.rand(n, T)
        
    elif media == 'uniform_smallvar':
        Z = 0.2*(-1 + 2*np.random.rand(n, T))
        
    elif media == 'limited info':
        Z = -0.01 + 0.02*np.random.rand(n, T)
        
    elif media == 'partial info':
        Z = -1 + 2*scipy.stats.beta.rvs(1, 8, size=(n, T))
        
    elif media == 'biased info': 
        Z[:n//2] = -1 + 2*scipy.stats.beta.rvs(1, 8, size=(n//2, T))
        Z[n//2:] = -1 + 2*scipy.stats.beta.rvs(8, 1, size=(n//2, T))
        
    elif media == 'biased_extreme':
        Z[:n//2] = -1
        Z[n//2:] = 1
        
    elif media == 'biased_deterministic':
        Z[:n//2, ] = -1/2
        Z[n//2:, ] = 1/2
        
    elif media == 'biased':
        Z[:n//2, ] = np.random.normal(loc=-1/2, scale=0.1, size=(n//2, T))
        Z[n//2:, ] = np.random.normal(loc=1/2, scale=0.1, size=(n//2, T))
        
    elif media == 'biased_internal':
        Z[:n//2, ] = -np.abs(Q[:n//2, np.newaxis]) + np.random.normal(0, 0.1, size=(n//2, T))
        Z[n//2:, ] = np.abs(Q[n//2:, np.newaxis]) + np.random.normal(0, 0.1, size=(n//2, T))
        
    Z = np.clip(Z, -1, 1)
        
    return Z

# external influence 
def external_influence(n, T, c, d, A, media, Q):
    
    # Q = internal_opinions(n, internal) # simulate internal opinions
    Z = media_signals(n, T, media, Q) # simulate media signals
    W = np.zeros((n, T)) # initialize matrix of external influences
    D = np.sum(A, axis=0) # in-degrees
    
    for i in range(n):
        if D[i] == 0: # if no incoming neighbors
            W[i, :] = c*Q[i]+d*Z[i, :] 
        else:
            W[i, :] = d*Z[i, :] # otherwise, scale the media signals
            
    return W

def precompute_M_powers(n, B, kappa, T):
    
    M = np.zeros((2, 2))
    # Compute the matrix M
    M[0, 0] = (B[0,0]*kappa[0][0])/(B[0,0]*kappa[0][0]+B[0,n-1]*kappa[1][0])
    M[0, 1] = (B[0,n-1]*kappa[1][0])/(B[0,0]*kappa[0][0]+B[0,n-1]*kappa[1][0])
    M[1, 0] = (B[n-1,0]*kappa[0][1])/(B[n-1,0]*kappa[0][1]+B[n-1,n-1]*kappa[1][1])
    M[1, 1] = (B[n-1,n-1]*kappa[1][1])/(B[n-1,0]*kappa[0][1]+B[n-1,n-1]*kappa[1][1])
    
    # Compute the powers up to M^T
    M_powers = [np.linalg.matrix_power(M, s) for s in range(1, T)]
    
    return M_powers

def MFA_process(n, T, c, d, W, R0, media, B, kappa):
    
    MFA = np.zeros((n, T+1))
    MFA[:, 0] = R0
    
    M_powers = precompute_M_powers(n, B, kappa, T)
    
    bar_W = np.zeros(2)
    if media == 'biased_internal' or media == 'biased': 
        bar_W[0] = -d/2  
        bar_W[1] = d/2  
    elif media == 'biased_extreme':
        bar_W[0] = -d
        bar_W[1] = d
    
    # Precompute the contribution of the term M^s\bar{W}
    M_bar_W = np.array([M_powers[s-1] @ bar_W for s in range(1, T)])
    
    for k in range(1, T+1):
        term1 = np.sum([(1-c-d)**t * W[:, k-t-1] for t in range(k)], axis=0)
        term2 = np.zeros(n)
        if k >= 2:
            for t in range(1, k):
                for s in range(1, t+1):
                    a_st = math.comb(t, s) * (1-c-d)**(t-s) * c**s
                    term2 += a_st * (
                        M_bar_W[s-1, 0] * (np.arange(n) < n//2) + 
                        M_bar_W[s-1, 1] * (np.arange(n) >= n//2)
                    )
        term3 = (1-c-d)**k * R0
        
        MFA[:, k] = term1 + term2 + term3 
        
    return MFA

# compute the max difference between original opinion process and MFA
# for various density regimes and different media signals
def MFA_accuracy(n, c, d, T, B, kappa, internal, media_list, theta_n_list, num_simulations=10):
    
    results = []
    
    for theta_n in theta_n_list:
        for media in media_list:
            diffs = [] # initialize list that will contain the max differences between the two processes
            for _ in range(num_simulations):
                A = dSBM(n, kappa, theta_n) # generate dSBM
                C = C_weights(n, c, d, A) # generate weights C
                Q = internal_opinions(n, internal) # generate internal opinions
                R = 2*np.random.rand(n)-1 # generate initial opinions independently of everything else
                W = d*media_signals(n, T, media, Q)
                #W = external_influence(n, T, c, d, A, media, Q) # generate media signals
                
                # keep the trajectory of the mean-field process
                MFA_trajectory = MFA_process(n, T, c, d, W, R, media, B, kappa)
                
                # initialize the trajectory of the opinion process
                R_trajectory = np.zeros((n, T+1))
                R_trajectory[:, 0] = R
                
                D = np.sum(A, axis=0) # in-degrees
    
                for i in range(n):
                    if D[i] == 0: # if no incoming neighbors
                        W[i, :] += c*Q[i]
                
                # iterate the opinion recursion and keep all the intermediate values 
                for t in range(1, T+1):
                    R = np.dot(C, R) + W[:, t-1]
                    R_trajectory[:, t] = R
                
                # calculate the maximum absolute difference of the two processes
                max_diff = np.max(np.abs(R_trajectory - MFA_trajectory))
                diffs.append(max_diff) # add the difference to the diffs list
                
            average_diff = np.mean(diffs) # compute the average difference based on 10 simulations
            std_diff = np.std(diffs) # compute the standard deviation of the differences based on 10 simulations
            results.append((theta_n, media, average_diff, std_diff))
            
    return results 

np.random.seed(7)

n = 2000
kappa = [[10, 1], [1, 2]]
c = 0.3
d = 0.6
T = 20
B = B_weights(n)

internal = 'uniform'
media_list = ['uniform_smallvar', 'biased_internal', 'biased_extreme']
theta_n_list = [1, np.log(n), np.sqrt(n), n/10]
theta_n_labels = ['1', r'$\log(n)$', r'$\sqrt{n}$', r'$\frac{n}{10}$']

results = MFA_accuracy(n, c, d, T, B, kappa, internal, media_list, theta_n_list)

for result in results:
    print(f"Theta_n: {result[0]}, Media: {result[1]}, Avg Diff: {result[2]:.3f}")