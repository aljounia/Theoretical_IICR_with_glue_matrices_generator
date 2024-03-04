# -*- coding: utf-8 -*-
"""
Implementation of the Non Stationary Structured Coalescent (NSSC)
"""
import numpy as np
import bisect
from math import exp, isclose, e
from scipy import integrate
from mpmath import mp,mpf

class NSSC:
    """
    Non Stationary Structured Coalescent
    This class represents a generalization of the Structured Coalescent
    Markov Chain for Non Stationary structured populations.
    """

    def __init__(self, model_params, lineages_are_dist=False):
        """
        Create a Structured Coalescent Markov Chain model.
        i.e.
        - a list of Q-matrices based on the input parameters
        - a list of time values indicating the time when some parameter change,
        implying a change in the Q-matrix (the length of this list is equal to
        one minus the length of the list of Q-matrices)
        - the sampling
        model_params: dictionary,
            nbLoci: integer, how many independent loci to simulate
                    (not used here)
            samplingVector: list of integer, how many sequences to sample from
                            each deme
            scenario: list of dictionaries. Each dictionary contains:
                    'time': real, the time to start with the configuration
                            (from present to past) the first dictionary of
                             the list has always 'time': 0
                    'migMatrix': matrix of real, the migration rate from
                    deme i to deme j
                    'demeSizeVector': list of real, the size of each deme
        lineages_are_dist: indicates whether lineages are distinguishable
                           or not
        """
        sampling_vector = model_params['samplingVector']
        n = len(sampling_vector)
        if 2 in sampling_vector:
            l1a = sampling_vector.index(2)
            l1b = sampling_vector.index(2)
        else:
            lineages_pos = [i for i, x in enumerate(sampling_vector) if x == 1]
            l1a = lineages_pos[0]
            l1b = lineages_pos[1]
        if lineages_are_dist:
            initial_state_vect = np.zeros(n**2+1)
            initial_state_ix = l1a*n + l1b
            initial_state_vect[initial_state_ix] = 1
        else:
            initial_state_vect = np.zeros(int(0.5*n*(n+1) + 1))
            initial_state_ix = int(l1a*n - 0.5*l1a*(l1a-1) + (l1b-l1a))
            initial_state_vect[initial_state_ix] = 1
        Qmatrix_list = []
        migMatrix_list = []
        time_list = []
        for i in range(len(model_params['scenario'])):
            d = model_params['scenario'][i]
            time_list.append(np.real(d['time']))
            migMatrix_list.append(np.array(d['M']))
            # Create the matrix Q
            Q = self.createQmatrix(np.array(d['M']), np.array(d['c']), lineages_are_dist)
            Qmatrix_list.append(Q)
            eigenvalues = np.linalg.eigvals(Q)
            #print(eigenvalues)
            # Compute the eigenvalues and vectors for the diagonalization
        self.time_list = time_list
        self.migMatrix_list = migMatrix_list
        print('M', self.migMatrix_list)
        self.Qmatrix_list = Qmatrix_list
        print('Q',self.Qmatrix_list)
        self.glue_matrix_list = []
        for i in range(0,len(self.Qmatrix_list)-1):
            glu = self.create_glue_matrix(self.Qmatrix_list[i].shape[0],self.Qmatrix_list[i+1].shape[0],self.migMatrix_list[i],self.migMatrix_list[i+1])
            self.glue_matrix_list.append(glu)
        print('G',self.glue_matrix_list)
        self.initial_state_vect = initial_state_vect
        self.initial_state_ix = initial_state_ix
        self.create_cum_prods_list()

    def createQmatrix(self, migMatrix, demeSizeVector, lineagesAreDist=False):
        """
        Create a Q-matrix based on the migration Matrix and the size
        of each deme
        """
        migMatrix = 0.5 * np.array(migMatrix)
        n = migMatrix.shape[0]
        kIx = np.arange(0, n)
        if lineagesAreDist:
            Q = np.zeros((n**2 + 1, n**2 + 1))
            for i in range(n):
                for j in range(n):
                    iMigratesTo = kIx[kIx != i]
                    Q[n*i+j, n*iMigratesTo + j] = migMatrix[i, iMigratesTo]
                    jMigratesTo = kIx[kIx != j]
                    Q[n*i+j, n*i + jMigratesTo] = migMatrix[j, jMigratesTo]
                # Coalescence inside a deme
                Q[n*i+i, -1] = 1./demeSizeVector[i]
        else:
            sizeQ = int(0.5*n*(n+1) + 1)
            Q = np.zeros((sizeQ, sizeQ))
            d = {}
            k = 0
            for i in range(n):
                for j in range(i, n):
                    d[(i, j)] = k
                    k += 1
            for i in range(n):
                for j in range(i, n):
                    rowNumber = d[(i, j)]
                    iMigratesTo = kIx[kIx != i]
                    columnNumbers = [d[(min(l, j), max(l, j))]
                                     for l in iMigratesTo]
                    Q[rowNumber, columnNumbers] = migMatrix[i, iMigratesTo]
                    jMigratesTo = kIx[kIx != j]
                    columnNumbers = [d[(min(i, l), max(i, l))]
                                     for l in jMigratesTo]
                    Q[rowNumber, columnNumbers] = Q[rowNumber,
                                                    columnNumbers] +\
                        migMatrix[j, jMigratesTo]
                # Coalescence inside a deme
                Q[d[(i, i)], -1] = 1./demeSizeVector[i]
        # Add the values of the diagonal
        columNumbers = np.arange(Q.shape[0])
        for i in columNumbers:
            noDiagColumn = columNumbers[columNumbers != i]
            Q[i, i] = -sum(Q[i, noDiagColumn])
        return(Q)

    def find_neighbours(self,index_deme, M,n):
        """Returns the list of indices of neighbour islands."""
        list_neighbours = []
        for i in range(n):
            if M[index_deme,i] > 0:
                list_neighbours.append(i)
        return list_neighbours

    def find_proba(self, deme_i, deme_j, M,n):
        neighbours = self.find_neighbours(deme_i,M,n)
        sum_mig = 0
        for neighbour in neighbours:
            sum_mig += M[deme_i][neighbour]
        return M[deme_i][deme_j]/sum_mig


    def create_glue_matrix(self, n1, n2, M1, M2, only_neighbour = False):
        G = np.zeros((n1,n2))
        ndemes1 = len(M1)
        ndemes2 = len(M2)
        d1 = dict(zip([(i,j) for i in range(ndemes1)
                       for j in range(i,ndemes1)],
                      [l for l in range(ndemes1**2)]))
        d2 = dict(zip([(i,j) for i in range(ndemes2)
                       for j in range(i,ndemes2)],
                      [l for l in range(ndemes2**2)]))

        for key in d1.keys():
            if key[1] < ndemes2 :
                G[d1[key],d2[key]] = 1
            elif key[1] >= ndemes2:
                neighbours = self.find_neighbours(key[1],M1,ndemes2)
                if key[0] == key[1] :
                    for colNumber in neighbours:
                        index_colNumber = [i for i in d2 if ((i[1]==colNumber and (i[0] in neighbours)) or (i[0]==colNumber and (i[1] in neighbours)))]
                        for ind_col in index_colNumber:
                            if ind_col[0] != ind_col[1] :
                                proba_1 = self.find_proba(key[0],ind_col[0],M1,ndemes2)
                                proba_2 = self.find_proba(key[1],ind_col[1],M1,ndemes2)
                                proba_3 = self.find_proba(key[0],ind_col[1],M1,ndemes2)
                                proba_4 = self.find_proba(key[1],ind_col[0],M1,ndemes2)
                                G[d1[key],d2[ind_col]] = proba_1*proba_2 + proba_3*proba_4
                            else:
                                proba_1 = self.find_proba(key[0],ind_col[0],M1,ndemes2)
                                proba_2 = self.find_proba(key[1],ind_col[1],M1,ndemes2)
                                G[d1[key],d2[ind_col]] = proba_1*proba_2 
                elif key[0] >= ndemes2 and key[1] >= ndemes2:
                    neighbours_2 = self.find_neighbours(key[0],M1,ndemes2)
                    print(key,neighbours,neighbours_2)
                    for neighb_1 in neighbours:
                        for neighb_2 in neighbours_2:
                            proba_1 = self.find_proba(key[0],neighb_2,M1,ndemes2)
                            print(proba_1)
                            proba_2 = self.find_proba(key[1],neighb_1,M1,ndemes2)
                            print(proba_2)
                            if neighb_2 >= neighb_1:
                                G[d1[key],d2[(neighb_1,neighb_2)]] = proba_1*proba_2
                            else:
                                G[d1[key],d2[(neighb_2,neighb_1)]] = proba_1*proba_2
                            
                else:
                    
                    colNumbers = [(key[0],l) for l in neighbours if (l >= key[0] and l < ndemes2)]
                    colNumbers.extend([(l,key[0]) for l in neighbours if (l < key[0] and key[0] < ndemes2)])
                    for colNumber in colNumbers:
                        print(key[0],key[1],colNumber[0],colNumber[1])
                        if key[0] == colNumber[0]:
                            proba = self.find_proba(key[1],colNumber[1],M1,ndemes2)
                        else:
                            proba = self.find_proba(key[1],colNumber[0],M1,ndemes2)
                        G[d1[key],d2[colNumber]] = proba
        G[-1,-1] = 1
        return(G)

    def create_cum_prods_list(self):
        """
        Create the list of cumulative product of Pt and the
        list of diagonalized Q_matrices
        """
        diagonalizedQ_list = []
        cum_prods = [np.eye(len(self.initial_state_vect))]
        for i in range(len(self.Qmatrix_list)-1):
            (A, D, Ainv) = self.diagonalize_Q(self.Qmatrix_list[i])
            diagonalizedQ_list.append((A, D, Ainv))
            t = self.time_list[i+1] - self.time_list[i]
            exp_tD = np.diag(np.exp(t * np.diag(D)))
            P_delta_t = A.dot(exp_tD).dot(Ainv)
            if cum_prods[-1].shape[0] != P_delta_t.shape[0]:
                glue_mat = self.glue_matrix_list[i-1]
                print(i)
                cum_prods.append(cum_prods[-1].dot(glue_mat).dot(P_delta_t))
            else :
                cum_prods.append(cum_prods[-1].dot(P_delta_t))
        (A, D, Ainv) = self.diagonalize_Q(self.Qmatrix_list[-1])
        diagonalizedQ_list.append((A, D, Ainv))
        self.diagonalizedQ_list = diagonalizedQ_list
        self.cum_prods = cum_prods
        #print('cum',cum_prods)

    def diagonalize_Q(self, Q):
        # Compute the eigenvalues and vectors for the diagonalization
        eigenval, eigenvect = np.linalg.eig(Q)
        # Put the eigenvalues in a diagonal
        D = np.diag(eigenval)
        A = eigenvect
        Ainv = np.linalg.inv(A)
        return (A, D, Ainv)

    def exponential_Q(self, t, i):
        """
        Computes e^{tQ_i} for a given t.
        Note that we will use the stored values of the diagonal expression
        of Q_i. The value of i is between 0 and the index of the last
        demographic event (i.e. the last change in the migration rate).
        """
        (A, D, Ainv) = self.diagonalizedQ_list[i]
        exp_tD = np.diag(np.exp(t * np.diag(D)))
        return(A.dot(exp_tD).dot(Ainv))

    def evaluate_Pt(self, t):
        """
        Evaluate the transition semigroup at t.
        Uses previously computed values to speed up the computation.
        """
        # Get the left of the time interval that contains t.
        i = bisect.bisect_right(self.time_list, t) - 1
        P_deltaT = self.exponential_Q(t - self.time_list[i], i)
        if (self.cum_prods[i].shape[0] != P_deltaT.shape[0]) or (self.cum_prods[i].shape[1] != P_deltaT.shape[1]):
            glue_mat = self.glue_matrix_list[i-1]
            P_deltaT = glue_mat.dot(P_deltaT)
        return(self.cum_prods[i].dot(P_deltaT))


    def cdfT2(self, t):
        """
        Evaluates the cumulative distribution function of T2
        for the current model
        """
        Pt = self.evaluate_Pt(t)
        return(np.real(Pt[self.initial_state_ix, -1]))

    def pdfT2(self, t):
        """
        Evaluates the probability density function of T2
        for the current model
        """
        S0 = self.initial_state_ix
        # Get the time interval that contains t.
        i = bisect.bisect_right(self.time_list, t) - 1
        cumulPt = self.cum_prods[i]
        Q = self.Qmatrix_list[i]
        if cumulPt.shape[0] != Q.shape[0]:
            glue_mat = self.glue_matrix_list[i-1]
            Q = glue_mat.dot(Q)
        P_delta_t = self.exponential_Q(t - self.time_list[i], i)
        return(cumulPt.dot(Q).dot(P_delta_t)[S0, -1])

    def evaluateIICR(self, t):
        """
        Evaluates the IICR at time t for the current model
        t: list of values to evaluate the IICR
        """
        # Sometimes, for numerical
        # errors, F_x and f_x get negative values
        # or "nan" or "inf"
        # In any of these cases, a default value is returned
        F_x = np.ones(len(t))
        f_x = np.ones(len(t))
        quotient_F_f = np.ones(len(t))
        eigvals = list(np.linalg.eigvals(self.Qmatrix_list[-1]))
        print("eig",eigvals)
        while max(eigvals) >= 0:
            eigvals.remove(max(eigvals))
        plateau = (-1)/max(eigvals)
        print(plateau)
        prec_plateau = 1e-3
        F_x[0] = self.cdfT2(t[0])
        if not(0<=F_x[0]<=1): F_x[0]=1
        f_x[0] = self.pdfT2(t[0])
        if (f_x[0] < 1e-14) or (np.isinf(f_x[0])) or np.isnan(f_x[0]):
            f_x[0] = 1e-14
        quotient_F_f[0] = (1-F_x[0])/f_x[0]
        for i in range(1, len(t)):
            F_x[i] = self.cdfT2(t[i])
            f_x[i] = self.pdfT2(t[i])
            quot_F_f = (1-F_x[i])/f_x[i]
            if i>10 and ((abs(quot_F_f-plateau) < prec_plateau) or (np.allclose(quotient_F_f[i-10:i- 1],np.repeat(plateau,9)))):
                quotient_F_f[i] = plateau
            else:
                quotient_F_f[i] = quot_F_f
        mean_f_x = self.compute_mean(t)
        print("mean",mean_f_x)
        print("var",self.compute_var(t)-(mean_f_x**2))
        print("diff",1-self.compute_differences_between_pair(t))
        #return(np.true_divide(1-F_x,f_x))
        return(quotient_F_f)

    def compute_distance(self, x, y, Nref):
        """
        Compute the distance between the IICR of the model,
        scaled by Nref, and some psmc curve given by x, y
        """
        # We evaluate the theoretical IICR at points
        # (x[i] + x[i-1])*0.5/(2*Nref)
        points_to_evaluate = [(x[i] + x[i-1])*0.25/Nref
                             for i in range(1, len(x))]
        m_IICR = self.evaluateIICR(points_to_evaluate)
        distance = 0
        for i in range(len(m_IICR)):
            distance += (y[i] - Nref*m_IICR[i])**2
        return distance

    def compute_mean(self, t):
        f_x = np.ones(len(t))
        f_x[0] = self.pdfT2(t[0])
        if (f_x[0] < 1e-14) or (np.isinf(f_x[0])) or np.isnan(f_x[0]):
            f_x[0] = 1e-14
        for i in range(1, len(t)):
            f_x[i] = self.pdfT2(t[i])
            if (f_x[i] < 1e-14) or (np.isinf(f_x[i])) or np.isnan(f_x[i]):
                f_x[i] = f_x[i-1]
        return integrate.simps(t*f_x, t)

    def compute_var(self, t):
        f_x = np.ones(len(t))
        f_x[0] = self.pdfT2(t[0])
        if (f_x[0] < 1e-14) or (np.isinf(f_x[0])) or np.isnan(f_x[0]):
            f_x[0] = 1e-14
        for i in range(1, len(t)):
            f_x[i] = self.pdfT2(t[i])
            if (f_x[i] < 1e-14) or (np.isinf(f_x[i])) or np.isnan(f_x[i]):
                f_x[i] = f_x[i-1]
        return integrate.simps((t**2)*f_x, t)

    def compute_differences_between_pair(self, t, theta=0.001, k=2):
        f_x = np.ones(len(t))
        f_x[0] = self.pdfT2(t[0])
        if (f_x[0] < 1e-14) or (np.isinf(f_x[0])) or np.isnan(f_x[0]):
            f_x[0] = 1e-14
        for i in range(1, len(t)):
            f_x[i] = self.pdfT2(t[i])
            if (f_x[i] < 1e-14) or (np.isinf(f_x[i])) or np.isnan(f_x[i]):
                f_x[i] = f_x[i-1]
        return integrate.simps((e**(-2*t*theta))*f_x, t)

class Pnisland(NSSC):
    """
    Piecewise n-island model
    """

    def __init__(self, model_params):
        """
        Create a Piecewise n-island model.
        i.e.
        - a list of Q-matrices based on the input parameters
        - a list of time values indicating the time when some parameter change,
        implying a change in the Q-matrix (the length of this list is equal to
        one minus the length of the list of Q-matrices)
        - the sampling
        model_params: dictionary,
            nbLoci: integer, how many independent loci to simulate
                    (not used here)
            samplingVector: list of integer, how many sequences to sample from
                            each deme
            scenario: list of dictionaries. Each dictionary contains:
                    'time': real, the time to start with the configuration
                            (from present to past) the first dictionary of
                             the list has always 'time': 0
                    'n': the number of demes
                    'M': the gene flow. Note there is no need for a migration
                          matrix because all demes exchange migrants with
                          the same rate under this model. Backware migrations
                          of one lineage occur with rate M.
                    'c': the size of the demes. Note that all demes
                                  has the same size because of the symmetry
                                  of the n-island model.
        """
        sampling_vector = model_params['samplingVector']
        if 2 in sampling_vector:
            self.initial_state_vect = np.array([1, 0, 0]) # Same deme
            self.initial_state_ix = 0
        else:
            self.initial_state_vect = np.array([0, 1, 0]) # Different demes
            self.initial_state_ix = 1
        Qmatrix_list = []
        time_list = []
        for i in range(len(model_params['scenario'])):
            d = model_params['scenario'][i]
            time_list.append(np.real(d['time']))
            # Create the matrix Q
            M = d['M']
            n = d['n']
            c = 1.0 / d['c']
            Q = self.createQmatrix(n, M, c)
            Qmatrix_list.append(Q)
            # Compute the eigenvalues and vectors for the diagonalization
        self.time_list = time_list
        self.Qmatrix_list = Qmatrix_list
        print('Q',self.Qmatrix_list)
        self.create_cum_prods_list()

    def createQmatrix(self, n, M, c):

        Q = np.array([[-M-c, M, c],
                      [float(M)/(n-1), -float(M)/(n-1), 0],
                      [0, 0, 0]])
        return Q
