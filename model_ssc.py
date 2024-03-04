"""
Implementation of the Stationary Structured Coalescent (NSSC)
"""
import numpy as np
import bisect

class SSC:
    """
    Stationary Structured Coalescent
    This class represents a Structured Coalescent
    Markov Chain for structured populations.
    """

    def __init__(self, model_params, lineages_are_dist=False):
        """
        Create a Structured Coalescent Markov Chain model.
        i.e.
        - a Q-matrix based on the input parameters
        - the sampling
        model_params: dictionary,
            samplingVector: list of integer, how many sequences to sample from
                            each deme
            'M': matrix of real, the migration rate from deme i to deme j
            'c': list of real, the size of each deme
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
        Q = self.createQmatrix(np.array(model_params['M']), np.array(
                model_params['size']), lineages_are_dist)
        print("M",model_params['M'])
        self.Qmatrix = Q
        print("Q",self.Qmatrix)
        self.initial_state_vect = initial_state_vect
        self.initial_state_ix = initial_state_ix
        self.diagonalizedQ = self.diagonalize_Q(self.Qmatrix)

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

    def diagonalize_Q(self, Q):
        # Compute the eigenvalues and vectors for the diagonalization
        eigenval, eigenvect = np.linalg.eig(Q)
        # Put the eigenvalues in a diagonal
        D = np.diag(eigenval)
        A = eigenvect
        Ainv = np.linalg.inv(A)
        return (A, D, Ainv)

    def exponential_Q(self, t, i=0):
        """
        Computes e^{tQ_i} for a given t.
        Note that we will use the stored values of the diagonal expression
        of Q_i. The value of i is between 0 and the index of the last
        demographic event (i.e. the last change in the migration rate).
        """
        (A, D, Ainv) = self.diagonalizedQ
        exp_tD = np.diag(np.exp(t * np.diag(D)))
        return(A.dot(exp_tD).dot(Ainv))

    def evaluate_Pt(self, t):
        """
        Evaluate the transition semigroup at t.
        Uses previously computed values to speed up the computation.
        """
        # Get the left of the time interval that contains t.
        P_deltaT = self.exponential_Q(t, 0)
        return(P_deltaT)

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
        P_delta_t = self.exponential_Q(t, 0)
        return(self.Qmatrix.dot(P_delta_t)[S0, -1])

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
        eigvals = list(np.linalg.eigvals(self.Qmatrix))
        eigvals.remove(max(eigvals))
        plateau = (-1)/max(eigvals)
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
            if (f_x[i] < 1e-14) or (np.isinf(f_x[i])) or np.isnan(f_x[i]):
                f_x[i] = f_x[i-1]
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
