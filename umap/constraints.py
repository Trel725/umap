import numpy as np
import numba
from numba.experimental import jitclass

#
# There are 2 broad class of constraint:
#
#  i) UMAP constructor can take submanifold projection functions.
#     These operate independent of any particular data set,
#     and are not passed the dataset point number as their 1st function arg.
#        Ex. fn(pt), fn(pts), fn(pt,grad) or fn(pts,grads)
#            where pts, grads could be any number of points.
# ii) 'fit' constructor (perhaps 'fit_transform') can take dataset-specific
#     constraints.  These take a dataset index as a 1st argument.  They also
#     have a variant that may operate on the full point cloud at once.
#        Ex. fn(idx,pt), fn(pts), fn(idx,pt,grad) or fn(pts, grads)
#            where pts, grads match the number of points in the data set.
#
# Each constraint has a spec describing constraint data set up by constructors
# Constraints typically apply as low-dimensional embeddings are optimized.
#
# Using jitclass, many "usability features" of python are not available,
# so more care needs to be taken to construct with right-typed arguments.
# layouts.py requires in-place mods, so be careful about '=' operator.
#
# "Hard" constraints, enforcing an inequality or equality condition, are fairly
# straight-forward.  "Soft" constraints really behave as "auxiliary forces" and
# can be through of as additional "tendencies" or "hints" or "regressions".
#
# So Hard constraints will have projections of points, and of gradients onto
# the tangent plane; whereas Soft constraints will generally only affect
# gradients.  Soft constraints are more aking to "forces" than "constraints".
#
# For example, multiplying gradients by 0 or 1 is a hard constraint,
# nicely described by the usual view of a tangent space for gradients.
# For tangent space, we assume the points already satisfy the constraint.
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

# A very liberal constraint ...
constrain_lo = np.float32(-10.0)
constrain_hi = np.float32(+10.0)
assert( constrain_lo < constrain_hi )
_mock_los = np.full(2, constrain_lo, dtype=np.float64)
_mock_his = np.full(2, constrain_hi, dtype=np.float64)
_mock_ones = np.ones(2, dtype=np.float64)
_mock_zeros = np.zeros(2, dtype=np.float64)

# Evolve from @jitclass version in "constraints.py" to discrete functions,
# mirroring the approach of distances.py

# Here, instead of a class encapsulating all params, the constraint parameters
# are passed as a tuple (of constraint_kwds 

# The class methods become function name suffixes
#   class method                    suffix      args
#   ------------------------------  -------     ----
#   project_onto_constraint         _pt         idx, pt
#   project_rows_onto_constraint    _pts        pts
#   project_onto_tangent_space      _grad       idx, pt, grad
#   project_rows_onto_tangent_space _grads      grads
#    (maybe) fit_onto_constraint    _fit        pts (?)
# for dataset-agnostic projectors (supplied to UMAP constructor) the
# 'idx' argument is dropped.

@numba.njit()
def noop_pt(idx,pt):
    return pt
@numba.njit()
def noop_pts(pts):
    return pts
@numba.njit()
def noop_grad(idx,pt,grad):
    return grad
@numba.njit()
def noop_grads(pts,grads):
    return grads

# New:  I see "inplace"-ness of operations is fragile in numba.
#       It can vary depending on whether caller is jitted or not,
#       and according to whether an argument is an array'view'
#
#       Try to make things not-in-place, from now on, for reliability
#       Never assume in-place modification is going to "just work".

@numba.njit()
def dimlohi_pt(pt, los=_mock_los, his=_mock_his):
    for i in range(los.size):
        if los[i] <= his[i]:
            if  pt[i] < los[i]:
                pt[i] = los[i]
            elif pt[i] > his[i]:
                 pt[i] = his[i]
    return pt

@numba.njit()
def dimlohi_pts(pts, los=_mock_los, his=_mock_his):
    assert len(pts.shape) == 2 and pts.shape[1] >= los.size and his.size==los.size
    for j in range(los.size):
        if los[j] <= his[j]:
            for i in range(pts.shape[0]):
                if  pts[i,j] < los[j]:
                    pts[i,j] = los[j]
                elif pts[i,j] > his[j]:
                     pts[i,j] = his[j]
    return pts

# dimlohi does not need tangent space (gradient) projection
# dimlohi_grad
# dimlohi_grads
# dimlohi_fit     (maybe)

#-------------- pinindexed --------------- certain vectors are fully pinned
@numba.njit()
def pinindexed_pt(idx, pt, pin_idx=None, pin_pos=None):
    """ pin_idx and pin_pos MUST be sorted by increasing pin_idx by caller.

        order = np.arsort(idx)
        pin_idx = pin_idx[order]
        pin_pos = pin_pos[order]
    """
    i = np.searchsorted( pin_idx, idx )         # binary search (uggh)
    if i < pin_idx.size and pin_idx[i] == idx:
        pt[:] = pin_pos[i,:]
        #print("pinindexed pt: i",i,"-->", pt)
    #else:
    #    print("pinindexed pt: i",i,"--> no-op")
    return pt

@numba.njit()
def pinindexed_pts(pts, pin_idx=None, pin_pos=None):
    """ pts is expected to be the full cloud of points,
        so all values of pin_idx are valid.
        (This is more efficient, no search)
    """
    assert pin_idx is not None and pin_pos is not None
    assert pts.shape[0] == pin_pos.shape[1]
    #print("pinindexed_pts: pin_idx=\n",pin_idx)
    #print("pinindexed_pts: pts=\n",pts)
    #print("pinindexed_pts: pts[pin_idx,:]=\n",pts[pin_idx,]) # "Command terminated"
    pts[pin_idx,] = pin_pos
    return pts

@numba.njit()
def pinindexed_grad(idx, pt, grad, pin_idx=None):
    i = np.searchsorted( pin_idx, idx )         # binary search (uggh)
    if i < pin_idx.size and pin_idx[i] == idx:
        grad[:] = 0.0
    return grad

@numba.njit()
def pinindexed_grads(pts, grads, pin_idx=[{}]):
    grads[pin_idx,:] = 0.0
    return grads

# --------------- freeinf ------------ np.inf are "free", all else pinned
# infs is same size as full point cloud, nsamp x dim
@numba.njit()
def freeinf_pt(idx, pt, infs):
    # idx must be a scalar :(
    assert infs is not None
    assert( infs.shape[1] == pt.shape[0] )
    if idx >= 0 and idx < infs.shape[0]:
        #vals = infs[idx,:].astype(pt.dtype)
        vals = infs[idx,:]         # assuming same dtype for pt, infs
        pt0 = pt.copy()
        pt[:] = np.where( ~np.isinf(vals), vals, pt )
        #if idx==13 or idx==14:
        #    print("freeinf_pt",idx,pt0,"-->",pt, infs[13],infs[14])
    return pt


@numba.njit()
def freeinf_pts(pts, infs=None):
    assert infs is not None
    assert pts.shape == infs.shape
    #pts = np.where( np.isinf(infs), pts, infs ) # issues when called from njit !?
    # if infs was float64, by mistake, ...
    #pts = np.where( np.isinf(infs), pts, infs ).astype(pts.dtype)
    #return pts
    pts[:,:] = np.where( np.isinf(infs), pts, infs )  # inf --> unchanged pts value
    return pts

@numba.njit()
def freeinf_grad(idx, pt, grad, infs=None):
    if infs is not None:
        if idx < infs.shape[0]:
            vals = infs[idx,:]
            mask = ~np.isinf(vals)
            grad[mask] = grad.dtype.type(0.0)
    return grad

@numba.njit()
def freeinf_grads(pts, grads, infs=None):
    #assert infs is not None
    # Here's how to match the scalar type with 'grads.dtype'
    grads[:,:] = np.where( np.isinf(infs), grads, grads.dtype.type(0.0) )
    return grads

# ----------------- springindexed ------------- indexed spring constants
# pin_idx ascending (re-sort pin_pos and springs to match)
# Note: these "springs" have equilibrium length zero!
@numba.njit()
def springindexed_pt(idx, pt, pin_idx=None, pin_pos=None, springs=None):
    assert pin_idx is not None and pin_pos is not None and springs is not None
    assert len(pin_idx.shape)==1 and len(pin_pos.shape)==2
    assert pin_idx.shape[0] == pin_pos.shape[0] and pin_idx.shape == springs.shape
    assert len(pt.shape)==1 and pt.shape[0] == pin_pos.shape[1]
    i = np.searchsorted(pin_idx, idx)
    if i < pin_idx.size and pin_idx[i] == idx and springs[i] == np.inf:
        pt[:] = pin_pos[i]
    return pt

@numba.njit()
def springindexed_pts(pts, pin_idx=None, pin_pos=None, springs=None):
    assert pin_idx is not None and pin_pos is not None and springs is not None
    assert len(pin_idx.shape)==1 and len(pin_pos.shape)==2
    assert pin_idx.shape[0] == pin_pos.shape[0] and pin_idx.shape == springs.shape
    pts[ pin_idx[np.isinf(springs)], : ] = pin_pos[ np.isinf(springs) ]
    return pts

@numba.njit()
def springindexed_grad(idx, pt, grad, pin_idx=None, pin_pos=None, springs=None):
    i = np.searchsorted(pin_idx, idx)
    if i < pin_idx.size and pin_idx[i] == idx:
        if springs[i] == np.inf:
            #pt[:] = pin_pos[i]  (assume this holds)
            grad[:] = 0.0
        else:
            delta = pin_pos[i] - pt
            #dist2 = np.sum(np.square(delta))
            #dist = np.sqrt(dist2)
            dist = np.sqrt(np.sum(np.square(delta)))
            #corr = (springs[i] * dist2) * (delta / np.sqrt(dist2))
            #corr = (springs[i] * dist) * delta
            #corr = np.where( corr > 4.0, 4.0,
            #                np.where( corr < -4.0, -4.0, corr))
            #grad[:] += corr
            grad += (springs[i] * dist) * delta
    return grad

@numba.njit()
def springindexed_grads(pts, grads, pin_idx=None, pin_pos=None, springs=None):
    for (i,idx) in enumerate(pin_idx):
        if springs[i] == np.inf:
            # 0.0 (don't care) assumes we'll project pt after applying grad
            # alt is to cap springs[i] and have a real force
            grads[idx,:] = 0.0
        else:
            delta = pin_pos[i] - pts[idx,:]
            dist = np.sqrt(np.sum(np.square(delta)))
            grads[idx,:] += (springs[i] * dist) * delta
    return grads

def test_dimlohi():
    x = np.empty( (8,8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            x[i,j] = 8.0*i + j
    los = np.empty((6), dtype=np.float32)
    his = np.empty((6), dtype=np.float32)
    # pin first 6 dims progressively tighter
    # last dim (5) pinned to exactly 50.0
    # higher dims (6,7) -> no-op
    for d in range(6):
        los[d] = 10.*d
        his[d] = 100. - 10.*d
    tupargs = (los, his)
    #print(los)
    #print(his)

    # test 1 : project individual [non-indexed] points
    #print(x)
    xcopy = x.copy();
    x2 = dimlohi_pt( x[2,:], *tupargs )
    #print(x2)
    for d in range(8):
        #print("d",d,"x2",x2[d])
        if d < 6:
            assert( x2[d] >= los[d] )
            assert( x2[d] <= his[d] )
        else:
            assert( x2[d] == 2*8 + d )

    # test 1 : project full cloude
    #print(x)
    xcopy = x.copy();
    x2 = dimlohi_pts( xcopy, *tupargs )
    #print(x2)
    assert np.all(x2 == xcopy) # non-view is modified in-place
    for i in range(8):
        for d in range(8):
            if d < 6:
                assert( x2[i,d] >= los[d] )
                assert( x2[i,d] <= his[d] )
            else:
                assert( x2[i,d] == i*8 + d )
    print("test dimlohi OK")

def test_pinindexed():
    # In this test, many things OK without numba create problems when jitted
    #   Ex. "in-place" operations often did not work.
    #       dictionary values (unsafe) caused garbage values
    x = np.empty( (8,8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            x[i,j] = 8.0*i + j
    x_ind = np.array([0,3,6])
    x_pos = np.vstack((np.repeat(-1.0, 8),
                       np.repeat(-2.0, 8),
                       np.repeat(-3.0, 8))).astype(np.float32)
    #print("x_ind",x_ind)
    #print("x_pos",x_pos)

    x0 = x[0,:]
    #print("x0",x[0,:])
    #print("x0",x0)
    y = pinindexed_pt(0, x[0,:].copy(), x_ind.copy(), x_pos.copy())
    #print("x_ind 0",x_ind)
    #print("x_pos 0",x_pos)
    #print("x0",x[0,:])
    #print("y0",y)
    assert np.all(y == -1.0)
    #
    # dictionary values --> tuple conversion DID NOT WORK ...
    kwargs = {'pin_index': x_ind, 'pin_pos': x_pos }
    #tupargs = tuple(kwargs.values())
    #print("tupargs 1",tupargs)
    #y = pinindexed_pt(0, x[0,:].copy(), *tupargs)
    #print("tupargs 1'",tupargs)
    #print("x_ind 1",x_ind) # garbage !?
    #print("x_pos 1",x_pos)
    #print("y1",y)
    #assert np.all(y == -1.0)
    #
    # Instead, can be safer:
    tupargs = (kwargs['pin_index'], kwargs['pin_pos']) # more robust dict->tuple
    tupargs_grad = (kwargs['pin_index'],) # comma forces this to be list
    print("tupargs_grad",tupargs_grad)
    #  or simpler, supply tuple directly
    #tupargs = (x_ind, x_pos)
    x0 = x[0,:].copy()
    #print("x0",x0)
    #print("tupargs 2",tupargs) # garbage !?
    y = pinindexed_pt(0, x0, *tupargs)
    #print("x0'",x0)                 # no change?  NOT IN PLACE (surprise)
    #print("tupargs 2'",tupargs)
    #print("x_ind 2",x_ind) # <--- value has changed!
    #print("x_pos 2",x_pos)
    #print("y 2",y)
    assert np.all(y == -1.0)
    #  assert np.all(x0 == x[0,:])   # inplace did not happen (oops)
    assert np.all(x0 == x_pos[0,:])  # inplace happens

    #print("x_ind",x_ind)
    #print("x_pos",x_pos)
    y = pinindexed_pt(1, x[1,:].copy(), x_ind, x_pos)
    #print("y3",y)
    assert np.all(y[:] == x[1,:])
    y = pinindexed_pt(-1, x[1,:].copy(), x_ind, x_pos)
    #print("y4",y)
    assert np.all(y[:] == x[1,:]) # idx not found --> no-op
    y = pinindexed_pt(999, x[1,:].copy(), x_ind, x_pos)
    #print("y5",y)
    assert np.all(y[:] == x[1,:]) # idx not found --> no-op

    xcopy = x.copy()
    #y = pinindexed_pts( x.copy(), x_ind, x_pos )
    y = pinindexed_pts( xcopy, x_ind, x_pos )
    #print("y6",y)
    assert np.all( y[0,:] == -1.0 )
    assert np.all( y[1,:] == x[1,:] )
    assert np.all( y[2,:] == x[2,:] )
    assert np.all( y[3,:] == -2.0 )
    # ...
    assert np.all( y[6,:] == -3.0 )
    assert np.all( y[7,:] == x[7,:] )

    grads = np.ones( (8,8), dtype=np.float32 )
    y = pinindexed_grad(0, x[0,:], grads[0,:].copy(), *tupargs_grad)
    #print("y7",y)
    assert( np.all( y == 0.0 ))
    y = pinindexed_grad(-1, x[0,:], grads[0,:].copy(), *tupargs_grad)
    #print("y8",y)
    assert( np.all( y == grads[0,:]))

    y = pinindexed_grads( x, grads.copy(), x_ind ) # remove x_pos arg!
    #print("y10",y)
    assert np.all( y[0,:] == 0.0 )
    assert np.all( y[1,:] == 1.0 )
    assert np.all( y[2,:] == 1.0 )
    assert np.all( y[3,:] == 0.0 )
    # ...
    assert np.all( y[6,:] == 0.0 )
    assert np.all( y[7,:] == 1.0 )

    y = pinindexed_grads( x, grads, *tupargs_grad )
    assert np.all(grads == y) # in-place ? This time, yes
    assert np.all( y[0,:] == 0.0 )
    assert np.all( y[1,:] == 1.0 )
    assert np.all( y[2,:] == 1.0 )
    assert np.all( y[3,:] == 0.0 )
    # ...
    assert np.all( y[6,:] == 0.0 )
    assert np.all( y[7,:] == 1.0 )
    print("test pinindexed OK")

def test_freeinf():
    samp = 5
    dim  = 3

    # either type for data seems ok
    #data = np.random.rand(samp,dim).astype(np.float64)
    data = np.random.rand(samp,dim).astype(np.float32)

    #infs = np.ones((samp,dim), dtype=np.float32)
    #infs *= 13.0                       # infs remains float32
    # usually might write:
    infs = 13.0 * np.ones((samp,dim), dtype=np.float32)
    # Caution: changes infs to float64 (which causes numba issues later)

    free = [(0,0), (0,1), (0,2), (2,0), (2,2), (3,1)]
    for pos in free:
        infs[pos] = np.inf
    infs = infs.astype(np.float32)   # if you've been lax
    #
    # NOTE: A one-element tuple requires a comma at then end
    #
    #tupargs = (infs)  # NOTE: tuple does NOT work (it expands row-wise!)
    tupargs = (infs.astype(np.float32),)

    #print("data", "\n", data)
    #print("infs", "\n", infs)
    #print("tupargs",tupargs)
    copy = data.copy()
    pts = freeinf_pts( copy, *tupargs )
    #pts = freeinf_pts( data.copy(), infs )      # also OK
    #print("pts\n",pts)
    inflist = np.isinf(infs)
    # OK in python, but not numba
    #assert np.all( pts[ inflist ] == data[ inflist ] )
    #assert np.all( pts[~inflist ] == 1.0 )
    for i in range(pts.shape[0]):
        inf_i = np.isinf(infs[i])
        # [i,inf_i] doesn't work with numba, so...
        assert np.all( pts[i][inf_i] == data[i][inf_i] )
        assert np.all( pts[i][~inf_i] == 13.0 )
    pts = np.zeros_like(data, dtype=np.float32)
    grads = 13.0 * np.ones_like(data)
    grads = grads.astype(np.float32)
    copy = data.copy()
    grads = freeinf_grads( copy, grads, *tupargs )
    #print("grads\n",grads)
    for i in range(pts.shape[0]):
        inf_i = np.isinf(infs[i])
        assert np.all( grads[i][inf_i] == 13.0 )
        assert np.all( grads[i][~inf_i] == 0.0 )

    pts = np.zeros_like(data)
    grads = 13.0 * np.ones_like(data)
    grads = grads.astype(np.float32)
    for i in range(pts.shape[0]):
        #print("\ni",i,"pts[i,:]",pts[i,:])
        pts[i,:] = freeinf_pt(i, data[i,:].copy(), *tupargs)
        grads[i,:] = freeinf_grad(i, data[i,:].copy(), grads[i,:], *tupargs)
        inf_i = np.isinf(infs[i,:])
        #print("inf_i",inf_i)
        #print("   pts[i,:]",pts[i,:])
        #print("   pts[i,inf_i]", pts[i,inf_i])
        #print("   data[i,inf_i]", data[i,inf_i])
        assert np.all( pts[i][inf_i] == data[i][inf_i] )
        assert np.all( pts[i][~inf_i] == 13.0 )
        #print("i",i," grads_i",grads[i,:])
        assert np.all( grads[i][inf_i] == 13.0 )
        assert np.all( grads[i][~inf_i] == 0.0 )
    
    # try idx as vector...         did not work (2d array indexed by 2d bool unsupported)
    #pts = np.zeros_like(data)
    #idxs = np.array([0,2,3], dtype=np.int32)
    #pts[idxs,:] = freeinf_pt(idxs, data[idxs,:].copy(), *tupargs)
    #print("pts[i,:]",pts[i,:])

    #print("pts", "\n", pts)
    print("test freeinf OK")

def test_springindexed():
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
    # if this test is jitted, cannot call numba.typeof
    print("data\n",data)
    print("idx (anchors)\n", idx)
    print("force constants\n", force_constants)
    print("pos (anchor positions)\n", pos)

    #  idx[npin]
    #  force_constants[npin]
    #  pos[npin,dim]
    #spi = SoftPinIndexed(idx, force_constants, pos)
    tupargs = (idx, pos, force_constants) # must be this order
    gvec = np.zeros_like(pos, dtype=np.float32)
    for ii in range(len(idx)):
        i = idx[ii]
        #print("\nNow move pin #", i, "force_constant", force_constants[ii])
        ivec = data[i,].copy()
        #print("ivec init", type(ivec), ivec)
        #jvec = spi.project_index_onto_constraint( i, ivec )
        jvec = springindexed_pt(i, ivec, *tupargs)
        gvec[ii] = springindexed_grad(i, data[i,].copy(), gvec[ii], *tupargs)
        #print(numba.typeof(data[i,]), numba.typeof(gvec[ii]))
        print("\ni", i, "ivec",ivec)
        print("force",force_constants[ii])
        #print("data[i,]",data[i,])
        print("jvec",jvec)
        print("gvec",gvec[ii])
        # assert np.all(ivec == jvec) # can fail
        ray = pos[ii] - data[i,]
        ray2 = np.sum(np.square(ray))         # square of spring length
        rayforce = force_constants[ii] * ray2 # magnitude of force
        raylen = np.sqrt(ray2)                # spring length
        unitray = ray / raylen
        print("rayforce * unitray", rayforce * unitray)
        
        #data[i,] = jvec
        #print("data[i,]",data[i,])
        #
        # TODO: assertions!
        #
        if force_constants[ii] == np.inf:
            assert np.all(jvec == pos[ii])
            assert np.all(gvec[ii] == 0.0)   # convention for now
        else:
            assert np.all(jvec == ivec)
            assert np.sum(np.square(gvec[ii] - rayforce*unitray)) < 1.e-4
    # unindexed/invalid index:
    for i in [-1,0,1,999]:
        if i==0 or i==1:
            ivec = data[i,].copy()
        else:
            ivec = data[0,].copy()
        jvec = springindexed_pt(i, ivec.copy(), *tupargs)
        grad = 0.5 * np.ones_like(ivec, dtype=np.float32)
        gvec = springindexed_grad(i, ivec.copy(), grad, *tupargs)
        assert np.all( jvec == ivec )  # no pt   change, unindexed/invalid idx
        assert np.all( grad == 0.5 )   # no grad change, unindexed/invalid idx

    # springindexed_pts...
    # springindexed_grads...
    print("data\n",data)
    print("idx (anchors)\n", idx)
    print("force constants\n", force_constants)
    print("pos (anchor positions)\n", pos)
    pts = springindexed_pts( data.copy(), *tupargs )
    print("pts\n", pts)
    for i in range(data.shape[0]):
        ii = np.searchsorted(idx, i)
        if ii < idx.size and idx[ii] == i and force_constants[ii] == np.inf:
            assert np.all( pts[i] == pos[ii] )
        else:
            assert np.all( pts[i] == data[i] )
    grads = np.zeros_like(data, dtype=np.float32)
    grads = springindexed_grads( data.copy(), grads, *tupargs )
    for i in range(data.shape[0]):
        ii = np.searchsorted(idx, i)
        if ii < idx.size and idx[ii] == i and force_constants[ii] == np.inf:
            assert np.all( grads[i] == 0.0 )
        else:
            ray = pos[ii] - data[i,]
            ray2 = np.sum(np.square(ray))         # square of spring length
            rayforce = force_constants[ii] * ray2 # magnitude of force
            raylen = np.sqrt(ray2)                # spring length
            unitray = ray / raylen
            assert np.sum(np.square(grads[i] - rayforce * unitray)) < 1.e-4
    print("test springindexed OK")

if __name__ == "__main__":
    test_dimlohi()
    test_pinindexed()
    test_freeinf()
    test_springindexed()
    print("jitted test functions...")
    numba.njit(test_dimlohi)()
    numba.njit(test_pinindexed)()
    numba.njit(test_freeinf)()
    numba.njit(test_springindexed)()

#
