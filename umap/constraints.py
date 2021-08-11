import numpy as np
import numba
from numba.experimental import jitclass

constrain_lo = np.float32(-10.0)
constrain_hi = np.float32(+10.0)
assert( constrain_lo < constrain_hi )

dimLohiSpec = [
    ('lo',  numba.types.float32[:]),
    ('hi',  numba.types.float32[:]),
    #('size',numba.types.int32),,
]

@jitclass(dimLohiSpec)
class DimLohi(object):
    def __init__(self, lo, hi):
        """ clip dim[i] to range lo[i]..hi[i], for i with lo[i] < hi[i];
            o/w clip to [-10.0,+10.0].
        """
        if self.lo.size != self.hi.size:
            print("warning: DimLohi(lo[],hi[]) lo and hi vectors should have same size")
        #self.size = min(lo.size, hi.size)
        sz = min(lo.size, hi.size)
        self.lo = lo[0:sz]
        self.hi = hi[0:sz]
        for i in range(self.lo.size):
            #print("cmp",self.lo[i], self.hi[i])
            if self.lo[i] >= self.hi[i]:
                self.lo[i] = constrain_lo
                self.hi[i] = constrain_hi
        #print("DimLohi lo",self.lo, "hi", self.hi)

    #@property
    #def size(self):
    #    return self.lo.size

    #@numba.jit(numba.types.float32[:](numba.typeof(dimLohiSpec), numba.types.float32[:]))
    # --> "class members not yet supported"
    def project_onto_constraint(self, vec):
        """ In-place bounding of vec[] dimension-wise.
            
            Well, the first self.lo.size dimensions of vec[]).
            Ex. if lo,hi size is one, this constraint applies only to vec[0] "x".
        """
        if len(vec.shape) == 1 and vec.shape[0] >= self.lo.size:
            #nchange = 0
            for i in range(self.lo.size):
                #print("cmp",self.lo[i], vec[i], self.hi[i])
                if   vec[i] < self.lo[i]:
                     vec[i] = self.lo[i]
                     #nchange = nchange + 1
                elif vec[i] > self.hi[i]:
                     vec[i] = self.hi[i]
                     #nchange = nchange + 1
            #if nchange:
            #    print("DimLohi cmp",self.lo, vec[0:self.lo.size], self.hi)
        else:
            print("Mismatch between vec shape",vec.shape,"with DimLohi size", self.lo.size)
        return vec

    def project_rows_onto_constraint(self, mat):
        """ In-place bounding of mat[i,dim] dimension-wise """
        if len(mat.shape) == 2 and mat.shape[1] >= self.lo.size:
            for i in range(mat.shape[0]):
                for j in range(self.lo.size):
                    #print("cmp",self.lo[i], vec[i], self.hi[i])
                    if   mat[i,j] < self.lo[j]:
                         mat[i,j] = self.lo[j]
                    elif mat[i,j] > self.hi[j]:
                         mat[i,j] = self.hi[j]
        else:
            print("Mismatch between mat shape",mat.shape,"with DimLohi size", self.lo.size)
        return mat

    def fit_onto_constraint(self, mat):
        """ rescale point-cloud mat to (/almost) satisfy a constraint,
        while little (/no) change to relative distances.

        For example, a point cloud 'mat' may be translated, rotated and scaled
        to come "close" to satisfying the constraint.  This is particularly for
        initialization of the UMAP algorithm, where the spectral embedding may
        far exceed some scaling constraint.
        
        In this case `project_rows_onto_constraint` might be too severe, and
        destroy too much information in the 'mat' embedding.
        """
        # todo: (i) find a (smallest) scalar such that min-max dimension ranges
        # of every dimension lie within [lo,hi]
        # (ii) rescale and translate entire point cloud to center min-max of
        # each dim within [lo,hi]
        # We can assume that mat initial range is "reasonably" within -10,+10.
        #
        # Fancier: allow a rotation (PCA?) to align major axes sequential
        # with least-constrained (hi-lo) dimensions (rot+scale+translate).
        return mat

