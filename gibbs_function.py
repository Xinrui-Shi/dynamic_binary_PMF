# import packages

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import svds

# initialisation for node latent position
def init_xy(A, T, d):
    # A: if T is None, A is a single adjacency matrix; otherwise, A is a dictionary containing T adjacency matrices.
    # T: integer, number of matrices in A (T>=0) and t=0,1,...T-1
    # d: integer, dimension of latent positions

    for t in range(T):
        if t==0:
            A_average = A[0]
        else:
            A_average += A[t] 

    A_average = A_average/T

    x0,eigval_av,y0_T= svds(A_average,k=d)

    return abs(x0), abs(y0_T).T


# initialisation for indicator Q
def init_Q(A, T, M, N):
    # A: if T is None, A is a single adjacency matrix; otherwise, A is a dictionary containing T adjacency matrices.
    # T: integer, number of matrices in A (T>=0) and t=0, 1,...T
    # M: integer, number of source nodes
    # N: integer, number of destination nodes

    for t in range(T):
        # Q
        Q = np.zeros((M, T))
        Q[:,t] = np.squeeze(np.sum(A[t],axis=1))
        # Q_star
        Q_star = np.zeros((N, T))
        Q_star[:,t] = np.sum(A[t],axis=0)        
    
    Q[Q>0] = 1
    Q_star[Q_star>0] = 1

    return Q, Q_star
    
# simulatie zeta
def sim_zeta( a, b, c, x, M, d):
    # a: constant parameter
    # b: constant parameter
    # c: constant parameter
    # x: N*d-dimensional matrix
    # M: integer, number of source nodes
    # d: integer, dimension of latent position
    

    alpha = d * a + b
    zeta_star = []
    sum_xj = np.sum(x, axis=1)

    for i in range(M):
        zeta_star.append(np.random.gamma(shape = alpha, 
                                         scale = 1/(c + sum_xj[i]), 
                                         size = 1))

    return np.squeeze(np.array(zeta_star))

# sample x and y
def sim_xy(y, I, zeta, Q, Q_star, a, M, d):
    # y: N*d dimensional matrix, given latent positions
    # I: ancillary value of Z
    # zeta: N-dimensional array
    # Q: M*T dimensional matrix
    # Q_star: N*T dimensional matrix
    # a: parameter constant
    # M: integer, number of source nodes
    # d: integer, dimension of latent position

    x = np.zeros((M, d))
    QQ_star_y = Q @ (Q_star.T @ y)

    for i in range(M):
        # compute alpha parameter for Gamma distribution
        alpha = a + I[i,:]

        # compute beta parameter for Gamma distributionI
        beta = zeta[i] + QQ_star_y[i,:]

        # sample x_ir
        x[i, :] = np.random.gamma(shape = alpha, scale = 1/beta)
    
    return x

 # sample theta
def sim_theta(alpha, beta, Q, T, M):
    # a, b: constant, parameter of Beta distribution
    # Q: M*T dimensional matrix
    # T: integer, number of matrices in A (T>=0) and t=0, 1,...T
    # M: integer, number of source nodes
    # N: integer, number of destination nodes
    
    # compute sum of Q over T
    Q_sumT = np.sum(Q, axis=1)

    # compute parameters for beta distribution
    beta_a = alpha + Q_sumT
    beta_b = beta +T - Q_sumT

    # sample theta
    theta = np.random.beta(a = beta_a, b = beta_b, size=M)

    return theta

# compute sum of A_ijt over t
def sum_A(A, T, M, N):
    # A:  A is a dictionary containing T adjacency matrices.
    # T: integer, number of matrices in A (T>=0) and t=0, 1,...T
    # M: integer, number of source nodes
    # N: integer, number of destination nodes

    A_sum_ti = np.zeros((M, T))
    A_sum_tj = np.zeros((N, T))

    for t in range(T):
        A_sum_ti[:, t] = np.squeeze(np.sum(A[t], axis=1))
        A_sum_tj[:, t] = np.squeeze(np.sum(A[t], axis=0))
    
    return A_sum_ti, A_sum_tj

def sim_Q(A_sum_ti, x, y, Q_star, theta, M, T):
    # A_sum_ti: M*T -dimensional array, indicator for existence of the source node at time t
    # x: M*d dimensional matrix, given latent positions
    # y: N*d dimensional matrix, given latent positions 
    # Q_star: N*T dimensional matrix
    # theta: M-dimensional array, parameter of pior distribution for Q
    # M: integer, number of source nodes
    # N: integer, number of destination nodes
    # T: integer, number of matrices in A (T>=0) and t=0, 1,...T
    # d: integer, dimension of latent position

    Q = np.ones((M, T))

    xyQ_star = x @ (y.T @ Q_star)

    for i in range(M):
        # probability of existence
        p1 = 1 - theta[i]

        for t in range(T):
            if A_sum_ti[i,t] == 0:
                # probability of not existence
                p0 = theta[i] * np.exp(-xyQ_star[i,t])
                Q[i,t] = np.random.binomial(n=1, p = p0/(p1+p0), size=1)
    
    return Q

def sample_ztp(lam,upper_lim=20): 
    if lam > upper_lim:
        samp = 0
        while samp == 0:
            samp = np.random.poisson(lam)
        return samp
    else:

        k = 1
        try:
            t = np.exp(-lam) / (1 - np.exp(-lam)) * lam
        except RuntimeWarning:
             t = np.exp(-lam-1e-10) / (1 - np.exp(-lam-1e-10)) * (lam + 1e-10)
        s = t
        u = np.random.uniform()
        while s < u:
            k += 1
            t *= float(lam) / k
            s += t
        return k
    
def sim_N_I(x, y, M, N, T, d, A_dense):
    # x: M*d dimensional matrix, given latent positions
    # y: N*d dimensional matrix, given latent positions 
    # M: integer, number of source nodes
    # N: integer, number of destination nodes
    # T: integer, number of matrices in A (T>=0) and t=0, 1,...T
    # d: integer, dimension of latent position
    xy = x @ y.T

    # sample N, I and I_star
    N_mat = {}
    I_jr = np.zeros((N, d))
    I_ir_star = np.zeros((M, d))

    for t in range(T):
        N_mat[t] = np.zeros((M, N))
 

    for i in range(M):
        for j in range(N):
            pi_ij = x[i,:] * y[j,:]/xy[i,j]
            for t in range(T):
                if A_dense[t][i,j] >0:
                    N_mat[t][i, j] = sample_ztp(xy[i,j])
                    Z_ijt = np.squeeze(np.random.multinomial(
                                              n = N_mat[t][i,j], 
                                              pvals = pi_ij,
                                              size = 1))
                    I_jr[j,:] += Z_ijt
                    I_ir_star[i,:] += Z_ijt

    return N_mat, I_jr, I_ir_star

def convert_to_dense(A):

    lenth = len(A)
    dense_A = {}
    for l in range(lenth):
        dense_A[l] = A[l].todense()
    
    return dense_A

def log_likelihood_PMF(A, x, y, Q, Q_star, T, M, N):
    # A: if T is None, A is a single adjacency matrix; otherwise, A is a dictionary containing T adjacency matrices.
    # x: M*d dimensional matrix, given latent positions
    # y: N*d dimensional matrix, given latent positions
    # Q: M*T dimensional matrix
    # Q_star: N*T dimensional matrix
    # T: integer, number of matrices in A (T>=0) and t=0, 1,...T
    # M: integer, number of source nodes
    # N: integer, number of destination nodes

    value = 0
    
    for i in range(M):
        for j in range(N):
            inner_xy =  np.dot(x[i,:], y[j,:])
            for t in range(T):
                if A[t][i,j]>0:
                    value +=  np.log(np.exp(inner_xy)-1)
                    
    for t in range(T):
        QX = Q[0,t]*x[0,:]
        for i in range(1,M):
            QX += Q[i,t]*x[i,:]

        QY = Q_star[0,t]*y[0,:]
        for j in range(1,N):
            QY += Q_star[j,t]*y[j,:]

        value  -= np.dot(QX, QY)
    return value

# obtain negative class
def binary_class(A, M, N):

    negative_class = np.zeros((M*N,2))
    positive_class = np.zeros((M*N,2))
     
    neg_count = 0 
    pos_count = 0

    for i in range(M):   
        for j in range(N):
            if A[i,j] == 0:
                negative_class[neg_count,:] = np.array([i,j])
                neg_count += 1
            else:
                positive_class[pos_count,:] = np.array([i,j])
                pos_count += 1
            
    return positive_class[:pos_count,:],negative_class[:neg_count,:]


# compute probabilities for all possible links
def possible_link_prob(x_set, y_set, theta_set, theta_star_set,  M, N, warm_up=50):
    
    num = len(theta_set)
    
    link_probs = {}
    for i in range(M):
        for j in range(N):
            link_ij = np.array([theta_set[l][i]*theta_star_set[l][j]*
                      (1-np.exp(-np.dot(x_set[l][i,:],y_set[l][j,:]))) 
                      for l in range(1+warm_up,num)])
            link_probs[i,j] = np.mean(link_ij)
    
    return link_probs

def pfm_pred(positive_class, negative_class, link_probs, warm_up=50, return_label=True):
    # positive_class: counter
    # negative_class: counter
    # x: latent position, M*d
    # y: latent position, N*d
    

    ## Calculate the scores for negative_class
    scores_negative_class = []
    ## for pair in negative_class
    for pair in negative_class:
        scores_negative_class += [link_probs[int(pair[0]),int(pair[1])]]
        
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in positive_class:
        scores_positive_class +=  [link_probs[int(pair[0]),int(pair[1])]]
    #combine for x and y
    x = np.concatenate((np.array(scores_negative_class),np.array(scores_positive_class)))
    if return_label:
        y = np.concatenate((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
        return x,y
    else: 
        return x
    
# count frequency of existence of source node
def freq_node(dictionary_A, T, M, source_node=True):

    if source_node:
        sum_axis=1
    else:
        sum_axis=0

    source_node_count = np.zeros((1,M))

    for t in range(T):
        sum_col_A = np.squeeze(np.sum(dictionary_A[t], axis=sum_axis))
        source_node_count[sum_col_A>0] +=1
    
    return source_node_count

# filter out some given source node
def filter_out_class(dictionary_A, A, T, filter_out_node_idx, source_node=True):

    M, N= A.shape

    negative_class = np.zeros((M*N,2))
    positive_class = np.zeros((M*N,2))
     
    neg_count = 0 
    pos_count = 0


    # filter out required node:
    if source_node:
        picked_node_idx = np.linspace(1, M, M)
        logical_indicator = np.array([ idx not in filter_out_node_idx for idx in  picked_node_idx])
        picked_node_idx = picked_node_idx[logical_indicator]-1

        for i in picked_node_idx:   
            for j in range(N):
                if A[i,j] == 0:
                    negative_class[neg_count,:] = np.array([i,j])
                    neg_count += 1
                else:
                    positive_class[pos_count,:] = np.array([i,j])
                    pos_count += 1

    else:
        picked_node_idx = np.linspace(1, N, N)
        logical_indicator = np.array([ idx not in filter_out_node_idx for idx in  picked_node_idx])
        picked_node_idx = picked_node_idx[logical_indicator]-1

        for i in range(M):   
            for j in picked_node_idx:
                if A[i,j] == 0:
                    negative_class[neg_count,:] = np.array([i,j])
                    neg_count += 1
                else:
                    positive_class[pos_count,:] = np.array([i,j])
                    pos_count += 1
            
    return positive_class[:pos_count,:],negative_class[:neg_count,:]

# select some given source node
def select_idx_class(dictionary_A, A, T, picked_node_idx, source_node=True):

    M, N= A.shape

    negative_class = np.zeros((M*N,2))
    positive_class = np.zeros((M*N,2))
     
    neg_count = 0 
    pos_count = 0


    # select required node:
    if source_node:
        for i in picked_node_idx:   
            for j in range(N):
                if A[i,j] == 0:
                    negative_class[neg_count,:] = np.array([i,j])
                    neg_count += 1
                else:
                    positive_class[pos_count,:] = np.array([i,j])
                    pos_count += 1

    else:
        for i in range(M):   
            for j in picked_node_idx:
                if A[i,j] == 0:
                    negative_class[neg_count,:] = np.array([i,j])
                    neg_count += 1
                else:
                    positive_class[pos_count,:] = np.array([i,j])
                    pos_count += 1
            
    return positive_class[:pos_count,:],negative_class[:neg_count,:]

def select_mc(MC, idx):
    num_iteration = len(MC)

    select_mc = np.zeros((num_iteration,))

    if len(idx)==1:
        pos = int(idx)
        for l in range(num_iteration):
            select_mc[l] = MC[l][pos]

            
    if len(idx)==2:
        pos1, pos2 = int(idx[0]), int(idx[1])
        for l in range(num_iteration):
            select_mc[l] = MC[l][pos1,pos2]
    

    return select_mc
