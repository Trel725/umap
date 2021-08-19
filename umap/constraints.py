import numpy as np
import numba
from numba.experimental import jitclass

# A very liberal constraint ...
constrain_lo = np.float32(-5.0)
constrain_hi = np.float32(+5.0)
assert( constrain_lo < constrain_hi )

#
# Each constraint has a spec describing constraint data set up by constructors
# Constraints typically apply as low-dimensional embeddings are optimized.
#
# Using jitclass, many "usability features" of python are not available,
# so more care needs to be taken to construct with right-typed arguments.
#
# "Hard" constraints, enforcing an inequality or equality condition, are fairly
# straight-forward.  "Soft" constraints really behave as "auxiliary forces" and
# can be through of as additional "tendencies" or "hints" or "regressions".
#
# So Hard constraints will have projections of points, and of gradients onto
# the tangent plane; whereas Soft constraints will generally only affect
# gradients, bymodify only gradients.  Perhaps it's better to call them
# "forces" instead of "constraints".
#
# For example, multiplying gradients by 0 or 1 is a hard constraint,
# nicely described by the usual view of a tangent space for gradients.
#
# Multiplying gradients by float values is soft.
# Similarly, adding point-specific "spring" forces modifies the gradient,
# and again does not really evoke the usual idea of tangent space.
#

# Multiple constraints can be supplied as a dictionary:
# "grad"    HardPinIndexed, PinNoninf, SoftPinIndexed, Densmap
# "point"   DimLohi, (HardPinIndexed, PinNoninf)
# "cloud"   project_rows_onto_constraint PinNoninf,...
# 'epoch'
# 'pin'
#  ...

# For jit purposes, though, the constraint types must be fully known
# This causes some issues.  Easiest for now is to supply a single
# constraint argument of each type.   (Maybe provide a jit-compatible
# way to compose constraints, some day?)

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

            For now, this class can also fix a dimension to a constant value
            if lo==hi
        """
        if self.lo.size != self.hi.size:
            print("warning: DimLohi(lo[],hi[]) lo and hi vectors should have same size")
        #self.size = min(lo.size, hi.size)
        sz = min(lo.size, hi.size)
        self.lo = lo[0:sz]
        self.hi = hi[0:sz]
        # "fix" bad things (here's one way)
        for i in range(self.lo.size):
            if self.lo[i] > self.hi[i]:    # nonsense -> default bound
                # could also: swap, set both to avg value, ...
                self.lo[i] = constrain_lo
                self.hi[i] = constrain_hi
                #print("DimLohi lo",self.lo, "hi", self.hi)

    def name(self):
        return "DimLohi"

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
        # just for show, maybe useful for other 'whole point cloud' constraints
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

    def project_onto_tangent_space(self, vec, grad):
        """ Adjust grad to lie on tangent space at point vec.  Basically,
            take a close "allowed" directional derivative on the tangent
            plane and translate it to the origin (since zero must always
            be a member of the tangent space).

            vec is const and assumed to already satisfy the constraint.
            grad is modified in-place.
        """
        # At all points where 'project_onto_constraint' does something
        # (i.e. when at a boundary) the derivative must not push outward
        # from the boundary.
        if len(vec.shape) == 1 and vec.shape[0] >= self.lo.size and vec.shape==grad.shape:
            for i in range(self.lo.size):
                #print("cmp",self.lo[i], vec[i], self.hi[i], grad[i])
                if   vec[i] <= self.lo[i]: # strictly, well, "=="
                     grad[i] = np.max( grad[i], 0 ) # never make things worse
                elif vec[i] >= self.hi[i]: # pretend it's "=="
                     grad[i] = np.min( grad[i], 0 )
            #if nchange:
            #    print("DimLohi tan",self.lo, vec[0:self.lo.size], self.hi, grad[0:self.lo.size])
        else:
            print("project_onto_tangent_space mismatch between vec shape",
                  vec.shape, ", grad shape", grad.shape,
                  "with DimLohi size", self.lo.size)
        return grad

    def fit_onto_constraint(self, mat):
        """ rescale nsmaples x dim point-cloud mat to (/almost) satisfy a
        constraint, with little (/no) change to relative distances.

        The idea is to lessen the amount of readjustment required during
        subsequent iterations of layout.py

        For example, a point cloud 'mat' may be translated, rotated and scaled
        to come "close" to satisfying the constraint.  This is particularly
        for initialization of the UMAP algorithm, where 'init' mechanisms
        are oblivious of constraints.
        
        In this case `project_onto_constraint` might be too severe, and
        destroy too much information in the 'mat' embedding.  In this case,
        the first epochs may try wildly to adapt to a difficult constraint.
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

hardPinIndexedSpec = [
    ("idx",       numba.types.int32[:]),      # list of pinned samples
    ("npin",      numba.types.int32),         # len(idx)
    ("pos",       numba.types.float32[:,:]),  # positions of every pinned idx.  npin x lo-D
    #("tmpvec",    numba.types.float32[:]),
]

@jitclass(hardPinIndexedSpec)
class HardPinIndexed(object):
    """ Individual indexed vectors pinned (gradients multiplied by 0/1).

        Rows of point cloud matrix are vectors.
        This constraint keeps a given set of rows from moving.

        Project entire point cloud is most efficient.
        Projecting individual rows we need to search by index, and need to
        supply the sample # as an index.
    """
    def __init__(self, idx, pos):
        #idx = np.array(idx).astype(np.int32)    # not with jitclass
        #pos = np.array(pos).astype(np.float32)
        if not (len(idx.shape)==1 and
                len(pos.shape)==2 and
                idx.shape[0] == pos.shape[0]
                ):
            print("Warning: strange HardPinIndexed dims: idx",idx.shape,"  pos",pos.shape)
            #print("Warning: strange HardPinIndexed dims?")
        order = np.argsort(idx)   # array(int64,1d,C)
        self.idx = idx[order]
        self.pos = pos[order,]
        self.npin = len(self.idx)
        #self.tmpvec = np.repeat(0., self.pos.shape[1]).astype(np.float32)
        #if np.any(np.diff(self.idx) == 0): # not for jit
        if np.any( self.idx[:-1] == self.idx[1:]):
            print("Warning: duplicate indices in HardPinIndexed")

    def name(self):
        return f"HardPin[{len(self.idx)}]"

    def project_index_onto_constraint(self, idx, vec):
        """ If idx is anchored (idx in self.idx) return anchor
            (o.w. return vec).   In-place modification of vec,
            but only if vec is not a view/slice.   So safest is
            to NOT rely on in-place semantics (with numba)
        """
        i = np.searchsorted(self.idx, idx)
        if i < self.npin and self.idx[i] == idx:
            vec[:] = self.pos[i] # vec[:] for inplace modification
        return vec

    def project_index_onto_tangent_space(self, idx, vec, grad):
        """ zero out the gradients for vec, which must be one of the anchors """
        #del vec # not with jit
        i = np.searchsorted(self.idx, idx)
        if i < self.npin and self.idx[i] == idx:
            grad[:] = 0.0
        return grad

    def project_rows_onto_constraint(self, data):
        """ Take anchored (sample in self.idx) subset of data[samples, dim]
            to pinned posn.
        """
        data[self.idx,:] = self.pos
        return data

    def project_rows_onto_tangent_space(self, data, grads):
        """ zero out the gradients for vec, which must be one of the anchors """
        #del data # not with jit
        grads[self.idx,:] = 0.0
        return grads

# PinNoninf taking a numpy.ma MaskedArray (or matrix all non-inf entries "pinned"?)
# but keeping the indexed calls for individual row projections.
# This replaces searching for the index with a faster lookup.
pinNoninfSpec = [
    ("unpin",   numba.types.float32[:,:]),  # inf entries are unconstrained.
]

@jitclass(pinNoninfSpec)
class PinNoninf(object):
    """ matrix entries that are non-inf are pinned.

        Rows of point cloud matrix are vectors.
        This constraint keeps any non-inf matrix elements from moving.
        (a row can be partly constrained)

        Projecting individual rows, we supply the sample # as an index.
        Projections assume properly-sized input arguments.

        This should be a faster alternative to HardPinIndexed, because
        index lookups have been replaced with lookups.
    """
    def __init__(self, unpin):
        if not (len(unpin.shape)==2):
            print("Warning: strange PinNoninf shape:",unpin.shape)
        self.unpin = unpin

    def name(self):
        return f"HardPin[{unpin.shape}]"

    def project_index_onto_constraint(self, idx, vec):
        """ If row 'idx' of self.unpin is not infinity, change vec to self.unpin value.
            In-place modification of vec (if possible).
        """
        vals = self.unpin[idx,:]
        mask = ~np.isinf(vals)
        vec[mask] = vals[mask]
        return vec

    def project_index_onto_tangent_space(self, idx, vec, grad):
        """ zero out those gradients for vec
            that are non-inf in self.unpin[idx,].
        """
        #del vec # not with jit
        vals = self.unpin[idx,:]
        mask = ~np.isinf(vals)
        grad[mask] = 0.0
        return grad

    def project_rows_onto_constraint(self, data):
        """ Take anchored (sample in self.idx) subset of data[samples, dim]
            to pinned posn.
        """
        # 'where' faster than subarray assignment
        data = np.where( np.isinf(self.unpin), data, self.unpin )
        return data

    def project_rows_onto_tangent_space(self, data, grads):
        """ zero out the gradients for vec, which must be one of the anchors """
        #del data # not with jit
        #grads[self.idx,:] = 0.0
        grads = np.where( np.isinf(self.unpin), grads, 0 )
        return grads

softPinIndexedSpec = [
    ("idx",       numba.types.int32[:]),  # list of pinned samples, of length sum(is_pinned)
    ("npin",      numba.types.int32),         # len(idx)
    ("spring",    numba.types.float32[:]),# force constant.  (inf ~ pinned, 0 and nan
    ("pos",       numba.types.float32[:,:]),   # positions of every pinned idx.  npin x lo-D
]

@jitclass(softPinIndexedSpec)
class SoftPinIndexed(object):
    """ A *soft* constraint, mimicing springs attracting (repelling)
        certain (indexed) samples to fixed positions.
        
        This can mimic a "weighted regression" of some points toward preferred
        positions.

        Point projection is strange because it moves a point only for spring
        constant `np.inf`.  Tangent space "projection" mutates into just an
        attractive gradient adjustment to "bends towards" the target position.

        Todo: a variant with a repulsive radius, that adds a close-range
        repulsion of multiple inidices towards identical (or nearby) target
        points.
    """

    def __init__(self, idx, spring, pos):
        if not (len(idx.shape)==1 and
                len(pos.shape)==2 and
                idx.shape[0] == pos.shape[0]
                ):
            print("Warning: strange HardPinIndexed dims: idx",idx.shape,"  pos",pos.shape)
        order = np.argsort(idx)
        self.idx = idx[order]
        self.pos = pos[order,]
        self.npin = len(self.idx)
        #self.spring = np.broadcast_to( spring, self.idx.shape )
        #self.spring = np.zeros( self.npin )                #.astype(np.float32)
        #self.spring[:] = spring                                 # broadcast?
        self.spring = spring
        self.spring[ np.isnan(self.spring) ] = np.float32(0)
        #if np.any(np.diff(self.idx) == 0):
        if np.any( self.idx[:-1] == self.idx[1:]):
            print("Warning: duplicate indices in HardPinIndexed")

    def name(self):
        return f"SoftPinIndexed[{self.npin}]"

    def project_index_onto_constraint(self, idx, vec):
        i = np.searchsorted(self.idx, idx)
        if i < self.npin and self.idx[i] == idx and self.spring[i] == np.inf:
            vec = self.pos[i]
        return vec

    def project_index_onto_tangent_space(self, idx, vec, grad):
        """ zero out the gradients for vec, which must be one of the anchors """
        i = np.searchsorted(self.idx, idx)
        if self.idx[i] == idx:
            if self.spring[i] == np.inf:
                grad = 0.0 # or (more generally) "towards self.pos" ?
                # This kinda' assumes the point has already been properly
                # projected, as for "hard pinning case".   This needs some
                # sync with what's done client-side.
            else:
                grad += self.spring * np.square(vec - self.pos[i])
        return grad

    def project_rows_onto_constraint(self, data):
        """ Take infinite-force anchorings to pinned posn o/w no-op.  """
        moves = np.isinf[self.idx]
        data[ np.isinf[self.idx],: ] = self.pos
        return vec

    def project_rows_onto_tangent_space(self, data, grads):
        """ zero out the gradients for vec, which must be one of the anchors """
        #del data # not with jit
        grads[self.idx,:] = 0.0
        return grad


def test_HardPinIndexed():
    samp = 5
    dim  = 3

    # either type for data seems ok
    #data = np.random.rand(samp,dim).astype(np.float64)
    data = np.random.rand(samp,dim).astype(np.float32)

    # either is OK with python, not with jit
    #idx = [1,3] # python list
    idx = np.array([1,3], dtype=np.int32)

    #pos = np.vstack( (np.repeat(0.,dim), np.repeat(1.,dim)) )
    pos = np.vstack( (np.repeat(np.float32(0.),dim), np.repeat(np.float32(1.),dim)) )
    print("data\n",data)
    print("idx (anchors)\n", idx)
    print("pos (anchor positions)\n", pos)

    hpi = HardPinIndexed(idx,pos)
    i = idx[0]
    print("Now move pin #", i)
    print("data[i,]",data[i,])
    ivec = data[i,]
    #ivec = np.repeat(0.5, dim).astype(np.float32)
    print("ivec init", type(ivec), ivec)
    jvec = hpi.project_index_onto_constraint( i, ivec )
    if np.all(ivec == jvec):
        print("ivec changed inline")
    print("jvec",jvec)   # None (could not get return value to work
    # All 3 above were pinned to origin
    i = idx[1]
    print("\nNow move pin #", i)
    print("init data[i,]",data[i,])
    jvec = hpi.project_index_onto_constraint( i, data[i,] )
    print("jvec",jvec)
    if np.all(data[i,] == jvec):
        print("data[i,] changed inline")
    print("data[i,]",data[i,]) # this did NOT modify in-place
    data[i,] = jvec
    print("data[i,]",data[i,])

    # now with a different vec arg type (still works)
    ivec = np.repeat(13.0, dim).astype(np.float64)
    print("\nivec init", type(ivec), ivec)
    jvec = hpi.project_index_onto_constraint( i, ivec )
    if np.all(ivec == jvec):
        print("ivec changed inline")
    print("ivec",ivec)   # simple item IS changed in-place
    print("jvec",jvec)
    print("data[i,]",data[i,])

    #print a warning for crazy constructor
    posn = np.random.rand(9,9).astype(np.float32)
    hpi_bad = HardPinIndexed(idx,posn)

    # hard pinned index grad:
    gradients = 1.0 + np.int32(np.copy(data)*10)*0.1 # or whatever
    print("gradients init\n", gradients)
    dummy_vec = np.random.rand(dim)
    for i in range(samp):
        #print("i",i)
        #print("dummy_vec",dummy_vec)
        #print("grad i", gradients[i,])
        hpi.project_index_onto_tangent_space(i,dummy_vec,gradients[i,])
    print("gradients hard-pinned\n", gradients)
    assert(np.all(gradients[idx[0],:] == 0.0 ))
    assert(np.all(gradients[idx[1],:] == 0.0 ))
    print("project gradients individual indexed vectors OK")

    gradients = 1.0 + np.int32(np.copy(data)*10)*0.1 # or whatever
    # modify grads of full dataset:
    hpi.project_rows_onto_tangent_space(data, gradients) # data=don't care
    assert(np.all(gradients[idx[0],:] == 0.0 ))
    assert(np.all(gradients[idx[1],:] == 0.0 ))
    print("project gradients for entire gradient matrix OK")

    data2 = np.random.rand(5,3)
    print("\ndata init\n",data2)
    hpi.project_rows_onto_constraint( data2 )
    print("data\n",data2)
    assert(np.all(data2[idx[0],:] == pos[0])) # all-zeros
    assert(np.all(data2[idx[1],:] == pos[1])) # all-ones
    print("project full data onto constraint OK")


def test_DimLohi():
    x = np.arange(100, dtype=np.float32) # type important
    x.shape = (10,10)
    los = np.ndarray((6), dtype=np.float32)
    his = np.ndarray((6), dtype=np.float32)
    # pin first half of dims progressively tighter
    # dim (axis) 6 pinned to exactly 50.0
    for d in range(6):
        los[d] = 10.*d
        his[d] = 100. - 10.*d
    dlh = DimLohi( los, his )
    dlh.project_rows_onto_constraint(x)
    for i in range(10):
        for d in range(10):
            if d < 6:
                assert( x[i,d] >= los[d] )
                assert( x[i,d] <= his[d] )
            else:
                assert( x[i,d] == i*10 + d )
    print("test_DimLohi seems OK")

def test_SoftPinIndexed():
    samp = 5
    dim  = 3

    # either type for data seems ok
    #data = np.random.rand(samp,dim).astype(np.float64)
    data = np.random.rand(samp,dim).astype(np.float32)

    # either is OK with python, not with jit
    #idx = [1,3] # python list
    idx =             np.array([2,   3,   4     ], dtype=np.int32)
    force_constants = np.array([0.0, 1.0, np.inf], dtype=np.float32)
    # np.inf force_constant makes this data[4] hard-pinned.

    #pos = np.vstack( (np.repeat(0.,dim), np.repeat(1.,dim)) )
    pos = np.vstack((
        np.repeat(np.float32(-1.),dim),
        np.repeat(np.float32(0.0),dim),
        np.repeat(np.float32(1.),dim),
    ))
    print("data\n",data)
    print("idx (anchors)\n", idx)
    print("force constants\n", force_constants)
    print("pos (anchor positions)\n", pos)

    #  idx[npin]
    #  force_constants[npin]
    #  pos[npin,dim]
    spi = SoftPinIndexed(idx, force_constants, pos)
    for ii in range(len(idx)):
        i = idx[ii]
        print("\nNow move pin #", i, "force_constant", spi.spring[ii])
        ivec = data[i,]
        print("ivec init", type(ivec), ivec)
        jvec = spi.project_index_onto_constraint( i, ivec )
        print("ivec",ivec)   # simple item IS changed in-place
        print("jvec",jvec)
        print("data[i,]",data[i,])
        data[i,] = jvec
        print("data[i,]",data[i,])

    print("inf force on data[4,] -->",data[4,])
    assert( np.all(data[4,] == spi.pos[2,]) ) # only the inf force actually projects, for soft pinning
    print("SoftPinIndexed project_index_onto_constraint seems OK")

if __name__ == "__main__":
    test_DimLohi()
    test_HardPinIndexed()
    test_SoftPinIndexed()
    #test_PinNonif()  # TBD
#
