import numpy as np
import numba

#print("Let's demo python clustering internals (jit version)")
#print("This cell is the precursor a new file umap/constrain_clust.py\n")

#
# ---------------------- Functions --------------
# ouch.  these must be jittable now
# numba cannot handle a python list of numpy arrays.
# Let's break things apart to find numba-ready code blocks
#
@numba.njit()
def np_mean_axis_0(pts):
    cl_avg = np.zeros((pts.shape[1]), dtype=np.float32)
    for pt in range(pts.shape[0]):
        cl_avg += pts[pt,:]
    assert pts.shape[0] > 0
    cl_avg /= pts.shape[0]
    return cl_avg

#if False: # for reference
#    def spring_mv_v0_plain(pt, target, spring, lr=1.0, mindist=0.0):
#        vec = target - pt      # vector toward cluster center
#        vecnorm = np.linalg.norm(vec)
#        if mindist > 0.0:
#            bar = max(vecnorm - mindist, 0.0) # max move dist (along vec/vecnorm)
#            delta = (min(lr * spring, 1.0) * bar / vecnorm) * vec
#        else: # mindist == 0.0
#            delta = min(1.0, lr*spring) * vec
#        return delta

@numba.njit() # approx. void(i8, f4[:,:], i8[:], f8, f8)
def xdo_clustering_ipts_toward(idx, pts, target, lr, spring, mindist=0.0):
    """ Non-overshooting move of pts[idx,:] towards cofm(pts[cluster_pts]).

        lr: time step (1.0 will move exactly to equilibrium posn if spring==1)

        spring : spring constant "force ~ spring * displacement" (parabolic potential)

        mindist: to "not go all the way" towards target, but stop mindist away.
    """
    vec = target - pts[idx,:]      # vector toward cluster center
    vecnorm = np.linalg.norm(vec)
    if vecnorm > 1e-5:
        bar = max(vecnorm - mindist, 0.0) # max move dist (along vec/vecnorm)
        #delta = (min(lr * spring, 1.0) * bar / vecnorm) * vec
        mv = min(bar, lr * bar * spring)
        pts[idx,:] += (mv/vecnorm) * vec

    return

@numba.njit(
    locals={'tot_grad': numba.float32[:],
            'eqm_pos' : numba.float32[:],
            'sum_springs' : numba.float32,
            'n_springs'   : numba.int64,
            'spring'      : numba.float64
           }
)
def xdo_clustering_mult(idx, idx_cl, clusters, springs, pts):
    """ return target+grad info for idx w/ springs to  multiple clusters. """
    tot_grad = np.zeros(pts.shape[1],dtype=np.float32)
    eqm_pos  = np.zeros(pts.shape[1],dtype=np.float32)
    sum_springs = 0.0 #np.sum(np_springs)
    n_springs = 0

    # calculate equilibrium spring-weight target position
    #           and total spring-weighted gradient
    # every cluster exerts a force ind't of cluster size
    for cc in idx_cl:
        cluster_pts  = np.argwhere(clusters[cc,:]).flatten()
        # Note: 1. Each cluster centroid gets its spring regardless
        #          of how populated the cluster is.
        #       2. Size 1 cluster get included
        #       3. Spring constant doubles as weighting factor -- this
        #          might not hold for other spring force models!
        if len(cluster_pts) > 0:
            cl_avg = np_mean_axis_0( pts[cluster_pts,:] )
            # Now update sums for equilibrium posn and total gradient
            # springs -> force gradient -> non-overshooting 'delta' movement
            spring = springs[cc]
            sum_springs += spring
            n_springs += 1
            eqm_pos += spring * cl_avg
            vec = cl_avg - pts[idx,:]      # vector toward cluster center
            tot_grad += spring * vec       # only for kr^2 spring physics
    #
    eqm_pos /= sum_springs

    return (n_springs, eqm_pos, tot_grad)

@numba.njit()
def xdo_clustering_ipts(idx, pts, clusters, springs, lr=1.0, mindist=0.01):
    #assert len(pts.shape) == 2
    n_samples = pts.shape[0]
    n_clust = springs.shape[0] # it is a vector, one per cluster
    #  otherwise n_clust = np.max(clusters) + 1
    #n_clust = len(np_cluster_lists)
    #assert n_clust > 0
    #assert len(springs) == len(np_cluster_lists)
    #np_cluster_lists = [np.array(clist,np.int32) for clist in cluster_lists]
    #np_springs = np.array(springs)
    idx_in = clusters[:,idx]
    idx_cl = np.argwhere(idx_in).flatten()

    if len(idx_cl) < 1:
        return

    elif len(idx_cl) == 1:
        # Separate out an easy case (idx in single cluster)
        c = idx_cl[0]
        cluster_pts  = np.argwhere(clusters[c,:]).flatten()
        if len(cluster_pts) > 1: # cluster size 1? pts[idx] already at center!
            centroid = np_mean_axis_0( pts[cluster_pts,:] )  # cluster centroid
            spring = float(springs[c])
            lr = float(lr)
            xdo_clustering_ipts_toward(idx, pts, centroid,
                                       lr, spring, mindist=mindist)
        return

    #else: idx attracted to multiple clusters... len(idx_cl) > 1 ... rare?
    # jit test:
    (n_springs, eqm_pos, tot_grad) = xdo_clustering_mult(idx, idx_cl, clusters, springs, pts)
    
    if n_springs==0:  # all clusters empty? No-op
        return

    # Move toward centroid-of-clusters ignores mindist (moves "all the way")
    #    mindist=0 is easy to calculate
    vec = eqm_pos - pts[idx]      # vector toward spring-weighted equilibrium
    mv = lr * np.linalg.norm(tot_grad)
    #pts[idx,:] += mv * vec
    if mv >= 1.0:
        pts[idx,:] = eqm_pos
    else:
        pts[idx,:] += mv * vec
    
    return


@numba.njit()
def xdo_clustering_pts(pts, cluster2d, springs, lr=1.0, mindist=0.01):
    assert len(pts.shape) == 2
    n_samples = pts.shape[0]
    # one round of clustering every point once
    for idx in range(n_samples):
        xdo_clustering_ipts(idx, pts, cluster2d, springs, lr, mindist)
    return

#
# -------------- python (non-numba) --------------
# These show how to convert python args to expected numba-compliant types
#

# for completeness, since this might also become a 'mk_FOO' jit-fn-generator
def do_clustering_ipts_py(idx, pts, cluster_lists, springs, lr, mindist):
    """ python-ish front-end, with appropriate setup for just internals.

        pts: array[n_samples,dim]  sample ~ "idx"
        clusters: list-of-lists ~ (cluster, idx)
        springs: python list of numbers (np array[:]
        lr, mindist: python numbers
    """
    idx = int(idx)
    lr = float(lr)
    mindist = float(mindist)
    assert len(pts.shape) == 2
    assert len(cluster_lists) == len(springs)
    n_samples = pts.shape[0]
    assert idx < pts.shape[0]
    
    #np_cluster_lists = [np.array(clist,np.int32) for clist in cluster_lists]
    n_clust = len(cluster_lists)

    if n_samples==0 or n_clust==0:
        return
    
    #
    # generalization:  clusters[c,idx] is True IFF idx is in cluster c
    clusters = np.full((n_clust, n_samples), False, dtype=bool)
    for (c,members) in enumerate(cluster_lists):
        for m in members:
            clusters[c][m] = True
            
    springs = np.array(springs, dtype=np.float32)  # don't need python float64 default
    assert len(springs.shape) == 1
    assert springs.shape[0] > 0

    # no negative or inf or nan springs
    assert np.all(springs >= 0.0)  # actually nan is also NOT >= 0 so elision...
    assert np.count_nonzero((springs == np.inf) | (springs == np.nan)) == 0
    
    # pts is to be modified -- do not create a copy!
    
    # invoke jit fn (or create and return it)
    xdo_clustering_ipts( idx, pts, clusters, springs, lr=lr, mindist=mindist )

    return

# python "frontend-to-jit" demo for cell output
def do_clustering_pts_py(pts, cluster_lists, springs, lr, mindist):
    """ python-ish front-end, with appropriate setup for just internals.

        pts: array[n_samples,dim]  sample ~ "idx"
        clusters: list-of-lists ~ (cluster, idx)
        springs: python list of numbers
        lr, mindist: python numbers
    """
    lr = float(lr)
    mindist = float(mindist)
    assert len(pts.shape) == 2
    assert len(cluster_lists) == len(springs)
    n_samples = pts.shape[0]
    
    #np_cluster_lists = [np.array(clist,np.int32) for clist in cluster_lists]
    n_clust = len(cluster_lists)
    
    if n_samples==0 or n_clust==0:
        return
    
    clusters = np.full((n_clust, n_samples), False, dtype=bool)
    for (c,members) in enumerate(cluster_lists):
        for m in members:
            clusters[c][m] = True
            
    springs = np.array(springs, dtype=np.float32)  # don't need python float64 default
    # no negative or inf or nan springs
    assert np.all(springs >= 0.0)  # actually nan is also NOT >= 0 so elision...
    assert np.count_nonzero((springs == np.inf) | (springs == np.nan)) == 0
    
    # pts is to be modified -- do not create a copy!
    
    if False:
        print(f"{numba.typeof(pts)=} {pts.shape=}")
        print(f"{numba.typeof(clusters)=}")
        print(f"{numba.typeof(springs)=}")
        print(f"{numba.typeof(lr)=}")
        print(f"{numba.typeof(mindist)=}")
    
    # invoke jit fn (or create and return it)
    xdo_clustering_pts( pts, clusters, springs, lr=lr, mindist=mindist )
    
    return

#
# ----------------- do_FOO_py --> mk_FOO jit-generators -----------------
#

def mk_clustering_ipts(idx, pts, cluster_lists, springs, lr=1.0, mindist=0.01):
    """ python-ish front-end, with appropriate setup for just internals.

        pts: array[n_samples,dim]  sample ~ "idx"
        clusters: list-of-lists ~ (cluster, idx)
        springs: python list of numbers (np array[:]
        lr: python number
        mindist: python number (not yet used, supposed to be "close enough" radius)
        
        Return a jitted function referencing our local state vars
    """
    idx = int(idx)
    lr = float(lr)
    mindist = float(mindist)
    assert len(pts.shape) == 2
    assert len(cluster_lists) == len(springs)
    n_samples = pts.shape[0]
    assert idx < pts.shape[0]
    
    #np_cluster_lists = [np.array(clist,np.int32) for clist in cluster_lists]
    n_clust = len(cluster_lists)

    if n_samples==0 or n_clust==0:
        return
    
    #
    # generalization:  clusters[c,idx] is True IFF idx is in cluster c
    clusters = np.full((n_clust, n_samples), False, dtype=bool)
    for (c,members) in enumerate(cluster_lists):
        for m in members:
            clusters[c][m] = True
            
    springs = np.array(springs, dtype=np.float32)  # don't need python float64 default
    # no negative or inf or nan springs
    assert np.all(springs >= 0.0)  # actually nan is also NOT >= 0 so elision...
    assert np.count_nonzero((springs == np.inf) | (springs == np.nan)) == 0
    
    # pts is to be modified -- do not create a copy!
    
    @numba.njit()
    def my_xdo_clustering_ipts(idx, pts):
        # invoke jit fn (or create and return it)
        xdo_clustering_ipts( idx, pts, clusters, springs, lr=lr, mindist=mindist )

    return my_xdo_clustering_ipts

# python "frontend-to-jit" demo for cell output
def mk_clustering_pts(pts, cluster_lists, springs, lr=1.0, mindist=0.01):
    """ python-ish front-end, with appropriate setup for just internals.

        pts: array[n_samples,dim]  sample ~ "idx"  (used for dims)
        clusters: list-of-lists ~ (cluster, idx)
        springs: python list of numbers
        lr: python number
        mindist: python number (not yet used, supposed to be "close enough" radius)
        
        Return a jitted function referencing our local (private?) state vars
    """
    lr = float(lr)
    mindist = float(mindist)
    assert len(pts.shape) == 2
    assert len(cluster_lists) == len(springs)
    n_samples = pts.shape[0]
    
    #np_cluster_lists = [np.array(clist,np.int32) for clist in cluster_lists]
    n_clust = len(cluster_lists)
    
    if n_samples==0 or n_clust==0:
        return
    
    clusters = np.full((n_clust, n_samples), False, dtype=bool)
    for (c,members) in enumerate(cluster_lists):
        for m in members:
            clusters[c][m] = True
            
    springs = np.array(springs, dtype=np.float32)  # don't need python float64 default
    # no negative or inf or nan springs
    assert np.all(springs >= 0.0)  # actually nan is also NOT >= 0 so elision...
    assert np.count_nonzero((springs == np.inf) | (springs == np.nan)) == 0
    
    # pts is to be modified -- do not create a copy!
    
    if False:
        print(f"{numba.typeof(pts)=} {pts.shape=}")
        print(f"{numba.typeof(clusters)=}")
        print(f"{numba.typeof(springs)=}")
        print(f"{numba.typeof(lr)=}")
        print(f"{numba.typeof(mindist)=}")
    
    @numba.njit()
    def my_xdo_clustering_pts(pts):
        # invoke jit fn (or create and return it)
        xdo_clustering_pts( pts, clusters, springs, lr=lr, mindist=mindist )
    
    return my_xdo_clustering_pts

if __name__ == "__main__":
    #TODO: get the "readonly" mk_FOO numba types,
    #      put back "tight" jit typespecs, and
    #      pre-compile module-level things
    #
    # This test mirrors non-jit (debuggable) code in examples/constraints.ipynb
    #
    # --------------------- Inputs -----------------
    #

    n_samples = 8
    cluster_lists = [[1,2,3], [2,7,6]]
    #           avg    2.0      5.0
    # each of the 2 clusters has a spring constant
    springs = np.array([0.8, 2.0], dtype=np.float32)
    lr = 1.0   # learning rate (gradient multiplier) for spring forces

    pts = np.ndarray((n_samples,2), dtype=np.float32)
    for i in range(n_samples):
        pts[i,:] = (float(i),float(i%3))

    mindist=0.2

    #
    # ------------------------ python/jit test -------------
    #

    #xdo_clustering_pts(pts, clusters, springs, lr=lr, mindist=0.2)
    do_clustering_pts_py(pts, cluster_lists, springs, lr=lr, mindist=0.2)

    print("final positions (one epoch)")
    for i in range(pts.shape[0]):
        print(f"{i=} pt = {pts[i,:]}")

    try:
        pts_ref
        assert np.allclose(pts, pts_ref)
        print("Good: matched pts_ref")
    except NameError:
        pass

    print("Goodbye!")
#
