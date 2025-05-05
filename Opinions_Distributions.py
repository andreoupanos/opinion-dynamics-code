import numpy as np
import random
import scipy.stats
from scipy.stats import beta, binom, expon, multivariate_normal
import seaborn as sns
import math
import matplotlib.pyplot as plt
import networkx as nx
from networkx import *
from scipy.stats import gaussian_kde
from scipy.linalg import fractional_matrix_power

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

# unnormalized weights
def B_weights(n):
    
    n1 = n2 = n//2 # number of nodes in each community 
    
    B = np.zeros((n, n))
    
    B[:n1, :n1] = 4  # community 1 to community 1
    B[:n1, n1:] = 1  # community 1 to community 2
    B[n1:, :n1] = 1  # community 2 to community 1
    B[n1:, n1:] = 4  # community 2 to community 2
    
    return B

# normalized weights
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

# internal opinions
def internal_opinions(n, l, internal):
    
    # n: number of nodes
    # l: number of topics
    
    n1 = n2 = n//2 # number of nodes in each community 
    
    Q = np.zeros((n, l)) # initialize matrix of internal opinions
    
    if internal == 'uniform':
        Q = 2*np.random.rand(n, l)-1 # Unif(-1, 1) for all nodes across all topics
        
    else:
        Q[:n1, :] = np.random.rand(n1, l)-1 # Unif(-1, 0) for nodes in community 1
        Q[n1:, :] = np.random.rand(n2, l) # Unif(0, 1) for nodes in communtiy 2
    
    return Q

# media signals
def media_signals(n, l, T, media, Q):
    
    # Q = internal_opinions(n, l, internal)
    
    Z = np.zeros((n, l, T)) # initialize the media tensor
    
    if media == 'uniform':
        Z = -1 + 2*np.random.rand(n, l, T)
        
    elif media == 'biased internal':
        for t in range(T):
            Z[:n//2, :, t] = np.random.normal(loc=-np.abs(Q[:n//2, :]), scale=0.1, size=(n//2, l))
            Z[n//2:, :, t] = np.random.normal(loc=np.abs(Q[n//2:, :]), scale=0.1, size=(n//2, l))
        
    elif media == 'both': # polarizing on 1st topic and uniform across the rest of topics
        for t in range(T):
            Z[:n//2, 0, t] = np.random.normal(loc=-np.abs(Q[:n//2, 0]), scale=0.1, size=(n//2))
            Z[n//2:, 0, t] = np.random.normal(loc=np.abs(Q[n//2:, 0]), scale=0.1, size=(n//2))
            Z[:, 1:, t] = -1 + 2*np.random.rand(n, l-1)
            
    elif media == 'both_d': # polarizing on 1st topic and uniform across the rest of topics
        for t in range(T):
            Z[:n//2, 0, t] = np.random.normal(loc=-1/2, scale=0.1, size=(n//2))
            Z[n//2:, 0, t] = np.random.normal(loc=1/2, scale=0.1, size=(n//2))
            Z[:, 1:, t] = -1 + 2*np.random.rand(n, l-1)
            
    Z = np.clip(Z, -1, 1)
            
    return Z

# external influence
def external_influence(n, l, T, c, d, A, media, Q):
    
    # Q = internal_opinions(n, l, internal)
    Z = media_signals(n, l, T, media, Q)
    W = np.zeros((n, l, T))
    D = np.sum(A, axis=0)
    
    for i in range(n):
        if D[i] == 0:
            W[i, :, :] = d * Z[i, :, :] + np.repeat(c*Q[i, :][:, np.newaxis], T, axis=1)
        else:
            W[i, :, :] = d * Z[i, :, :]
            
    return W

# precompute powers of the SBM matrix M
def precompute_M_powers(n, B, kappa, T):
    
    M = np.zeros((2, 2))
    # Compute the matrix M
    M[0, 0] = (B[0,0]*kappa[0][0])/(B[0,0]*kappa[0][0]+B[0,n-1]*kappa[1][0])
    M[0, 1] = (B[0,n-1]*kappa[1][0])/(B[0,0]*kappa[0][0]+B[0,n-1]*kappa[1][0])
    M[1, 0] = (B[n-1,0]*kappa[0][1])/(B[n-1,0]*kappa[0][1]+B[n-1,n-1]*kappa[1][1])
    M[1, 1] = (B[n-1,n-1]*kappa[1][1])/(B[n-1,0]*kappa[0][1]+B[n-1,n-1]*kappa[1][1])
    
    # Compute the powers up to M^{T-1}
    M_powers = [np.linalg.matrix_power(M, s) for s in range(1, T)]
    
    return M_powers

# simulate the mean-field process
def MFA_process(n, l, T, c, d, W, R0, media, B, kappa):
    
    MFA = np.zeros((n, l, T+1))
    MFA[:, :, 0] = R0
    
    M_powers = precompute_M_powers(n, B, kappa, T)
    
    bar_W = np.zeros((2, l))
    if media == 'both' or 'both_d': 
        bar_W[0, 0] = -d/2  
        bar_W[1, 0] = d/2   
    
    # Precompute the contribution of the term M^s\bar{W}
    M_bar_W = np.array([M_powers[s-1] @ bar_W for s in range(1, T)])
    
    for k in range(1, T+1):
        term1 = np.sum([(1-c-d)**t * W[:, :, k-t-1] for t in range(k)], axis=0)
        term2 = np.zeros((n, l))
        if k >= 2:
            for t in range(1, k):
                for s in range(1, t+1):
                    a_st = math.comb(t, s) * (1-c-d)**(t-s) * c**s
                    term2 += a_st * (
                        M_bar_W[s-1, 0, :] * (np.arange(n) < n//2)[:, np.newaxis] + 
                        M_bar_W[s-1, 1, :] * (np.arange(n) >= n//2)[:, np.newaxis]
                    )
        term3 = (1-c-d)**k * R0
        
        MFA[:, :, k] = term1 + term2 + term3 
        
    return MFA


### -------------------------------------------
### Density plots for different density regimes
### -------------------------------------------

np.random.seed(112)

# Accuracy of the mean-field approximation in terms of stationary distribution
# 4 density regimes: 1, log(n), sqrt{n}, n/10

n = 2000 # number of individuals
T = 20 # number of iterations 
n1 = n2 = n//2 # number of nodes in each community
l = 2 # number of topics 
kappa = [[10, 1], [1, 2]]
c = 0.6
d = 0.2
B = B_weights(n)
internal = 'uniform'
media = 'both'


Q = internal_opinions(n, l, internal)
R0 = 2*np.random.rand(n, l)-1
# generate external influence once
W = d*media_signals(n, l, T, media, Q)
# we generate MFA values once, since they don't depend on the density
mfa = np.zeros((n, l)) 
MFA_trajectory = MFA_process(n, l, T, c, d, W, R0, media, B, kappa)
mfa = MFA_trajectory[:, :, T] # independent of density parameter



# -----
# Dense
# -----

theta_n = n/10

# initialize the n by l matrix that will end up containing the opinions at time T
original_dense = np.zeros((n, l))

A = dSBM(n, kappa, theta_n)
C = C_weights(n, c, d, A)

# create copies of initial opinions for each density regime
R_dense = np.copy(R0)
R_sqrt = np.copy(R0)
R_log = np.copy(R0)
R_sparse = np.copy(R0)

# adjust the external influence in the case of 0 in-degree nodes
D = np.sum(A, axis=0)
for i in range(n):
    if D[i] == 0:
        W[i, :, :] += np.repeat(c*Q[i, :][:, np.newaxis], T, axis=1)

# Trajectory of original opinion process (dense)
R_trajectory = np.zeros((n, l, T+1))
R_trajectory[:, :, 0] = R_dense

#W = external_influence(n, l, T, c, d, A, internal, media)

for t in range(1, T+1):
    R_dense = np.dot(C, R_dense) + W[:, :, t-1]
    R_trajectory[:, :, t] = R_dense
    
original_dense = R_trajectory[:, :, T]

# -----------
# Square root
# -----------

theta_n = np.sqrt(n)

original_sqrt = np.zeros((n, l))

A = dSBM(n, kappa, theta_n)
C = C_weights(n, c, d, A)

D = np.sum(A, axis=0)
for i in range(n):
    if D[i] == 0:
        W[i, :, :] += np.repeat(c*Q[i, :][:, np.newaxis], T, axis=1)

# Trajectory of original opinion process (square root)
R_trajectory = np.zeros((n, l, T+1))
R_trajectory[:, :, 0] = R_sqrt

for t in range(1, T+1):
    R_sqrt = np.dot(C, R_sqrt) + W[:, :, t-1]
    R_trajectory[:, :, t] = R_sqrt
    
original_sqrt = R_trajectory[:, :, T]

# -----------
# Semi-sparse
# -----------

theta_n = np.log(n)

original_semisparse = np.zeros((n, l))

A = dSBM(n, kappa, theta_n)
C = C_weights(n, c, d, A)

D = np.sum(A, axis=0)
for i in range(n):
    if D[i] == 0:
        W[i, :, :] += np.repeat(c*Q[i, :][:, np.newaxis], T, axis=1)

# Trajectory of original opinion process (semi-sparse)
R_trajectory = np.zeros((n, l, T+1))
R_trajectory[:, :, 0] = R_log

for t in range(1, T+1):
    R_log = np.dot(C, R_log) + W[:, :, t-1]
    R_trajectory[:, :, t] = R_log
    
original_semisparse = R_trajectory[:, :, T]

# ------
# Sparse
# ------

theta_n = 1

original_sparse = np.zeros((n, l))

A = dSBM(n, kappa, theta_n)
C = C_weights(n, c, d, A)

D = np.sum(A, axis=0)
for i in range(n):
    if D[i] == 0:
        W[i, :, :] += np.repeat(c*Q[i, :][:, np.newaxis], T, axis=1)

# Trajectory of original opinion process (sparse)
R_trajectory = np.zeros((n, l, T+1))
R_trajectory[:, :, 0] = R_sparse

for t in range(1, T+1):
    R_sparse = np.dot(C, R_sparse) + W[:, :, t-1]
    R_trajectory[:, :, t] = R_sparse
    
original_sparse = R_trajectory[:, :, T]


sns.set_theme(style="darkgrid")

# Plot everything together
fig, axs = plt.subplots(1, l, figsize=(14, 6))

for topic in range(l):
    
    # Estimate KDEs
    kde_mfa = gaussian_kde(mfa[:, topic])
    kde_original_dense = gaussian_kde(original_dense[:, topic])
    kde_original_sqrt = gaussian_kde(original_sqrt[:, topic])
    kde_original_semisparse = gaussian_kde(original_semisparse[:, topic])
    kde_original_sparse = gaussian_kde(original_sparse[:, topic])
    
    # Plot KDEs
    x_vals = np.linspace(-1, 1, 100)

    # original processes
    axs[topic].plot(x_vals, kde_original_sparse(x_vals), label=r'$\theta_n = 1$', color='orange', linestyle='-')
    axs[topic].plot(x_vals, kde_original_semisparse(x_vals), label=r'$\theta_n = \log(n)$', color='brown', linestyle='-')
    axs[topic].plot(x_vals, kde_original_sqrt(x_vals), label=r'$\theta_n = \sqrt{n}$', color='green', linestyle='-')
    axs[topic].plot(x_vals, kde_original_dense(x_vals), label=r'$\theta_n = \frac{n}{10}$', color='navy', linestyle='-')

    # MFA
    axs[topic].plot(x_vals, kde_mfa(x_vals), label='Mean-field', color='yellow', linestyle='--', linewidth=2)

    axs[topic].set_xlabel(f'Topic {topic+1} Opinion Value', fontsize=14)
    axs[topic].set_ylabel('Density', fontsize=14)
    axs[topic].set_title(f'Density Comparison for Topic {topic+1}', fontsize=16)
    axs[topic].legend(fontsize=12)
    axs[topic].grid(True)

    
plt.tight_layout()
plt.show()


### ---------------------------------------------
### Scatterplots of initial vs expressed opinions
### ---------------------------------------------

n = 2000 # number of individuals
n1 = n2 = n//2 # number of nodes in each community

l = 2 # number of topics 
kappa = [[10, 1], [1, 2]]

c = 0.6
d = 0.2

T = 20 # number of iterations 

theta_n = np.log(n)

internal = 'uniform'
media = 'both'
    
# Generate the graph, weights, internal opinions, and external influences
A = dSBM(n, kappa, theta_n)
C = C_weights(n, c, d, A)
Q = internal_opinions(n, l, internal)
R = 2*np.random.rand(n, l)-1
R_init = np.copy(R)
W = external_influence(n, l, T, c, d, A, media, Q)

# Initialize the trajectory storage 
R_trajectory = np.zeros((n, l, T+1))
R_trajectory[:, :, 0] = R

# Simulate the original opinion process
for t in range(1, T+1):
    R = np.dot(C, R) + W[:, :, t-1]
    R_trajectory[:, :, t] = R
    
    
# Plot initial vs. expressed opinions
sns.set_theme(style="darkgrid")

initial_com1_opinions = R_init[:int(n1), :] # initial opinions of community 1
initial_com2_opinions = R_init[int(n1):int(n1+n2), :] # initial opinions of community 2
expressed_com1_opinions = R[:int(n1), :] # expressed (stationary) opinions of community 1 
expressed_com2_opinions = R[int(n1):int(n1+n2), :] # expressed (stationary) opinions of community 2 

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# scatterplot for initial opinions 
axes[0].scatter(initial_com1_opinions[:, 0], initial_com1_opinions[:, 1], color='blue', label='Community 1')
axes[0].scatter(initial_com2_opinions[:, 0], initial_com2_opinions[:, 1], color='red', label='Community 2')
axes[0].set_xlabel('Opinions on topic 1')
axes[0].set_ylabel('Opinions on topic 2')
axes[0].set_title('Initial Opinions')
axes[0].legend()
axes[0].grid(True)

# Scatterplot for expressed opinions
axes[1].scatter(expressed_com1_opinions[:, 0], expressed_com1_opinions[:, 1], color='blue', label='Community 1')
axes[1].scatter(expressed_com2_opinions[:, 0], expressed_com2_opinions[:, 1], color='red', label='Community 2')
axes[1].set_xlabel('Opinions on topic 1')
axes[1].set_ylabel('Opinions on topic 2')
axes[1].set_title('Expressed Opinions')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
#plt.savefig('InternalvsExpressed_CorrelatedMedia_AbsoluteInternals')
plt.show()