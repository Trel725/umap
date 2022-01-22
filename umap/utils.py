# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause

import time
from warnings import warn

import numpy as np
import numba
from sklearn.utils.validation import check_is_fitted
import scipy.sparse

#@numba.njit
#def reposition(X, up: List[Tuple(int,Tensor)])

@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """A fast computation of knn indices.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor indices of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit("f4(i8[:])")
def tau_rand(state):
    """A fast (pseudo)-random number generator for floats in the range [0,1]

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random float32 in the interval [0, 1]
    """
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


@numba.njit()
def norm(vec):
    """Compute the (standard l2) norm of a vector.

    Parameters
    ----------
    vec: array of shape (dim,)

    Returns
    -------
    The l2 norm of vec.
    """
    result = 0.0
    for i in range(vec.shape[0]):
        result += vec[i] * vec[i]
    return np.sqrt(result)



@numba.njit(parallel=True)
def submatrix(dmat, indices_col, n_neighbors):
    """Return a submatrix given an orginal matrix and the indices to keep.

    Parameters
    ----------
    dmat: array, shape (n_samples, n_samples)
        Original matrix.

    indices_col: array, shape (n_samples, n_neighbors)
        Indices to keep. Each row consists of the indices of the columns.

    n_neighbors: int
        Number of neighbors.

    Returns
    -------
    submat: array, shape (n_samples, n_neighbors)
        The corresponding submatrix.
    """
    n_samples_transform, n_samples_fit = dmat.shape
    submat = np.zeros((n_samples_transform, n_neighbors), dtype=dmat.dtype)
    for i in numba.prange(n_samples_transform):
        for j in numba.prange(n_neighbors):
            submat[i, j] = dmat[i, indices_col[i, j]]
    return submat


# Generates a timestamp for use in logging messages when verbose=True
def ts():
    return time.ctime(time.time())


# I'm not enough of a numba ninja to numba this successfully.
# np.arrays of lists, which are objects...
def csr_unique(matrix, return_index=True, return_inverse=True, return_counts=True):
    """Find the unique elements of a sparse csr matrix.
    We don't explicitly construct the unique matrix leaving that to the user
    who may not want to duplicate a massive array in memory.
    Returns the indices of the input array that give the unique values.
    Returns the indices of the unique array that reconstructs the input array.
    Returns the number of times each unique row appears in the input matrix.

    matrix: a csr matrix
    return_index = bool, optional
        If true, return the row indices of 'matrix'
    return_inverse: bool, optional
        If true, return the the indices of the unique array that can be
           used to reconstruct 'matrix'.
    return_counts = bool, optional
        If true, returns the number of times each unique item appears in 'matrix'

    The unique matrix can computed via
    unique_matrix = matrix[index]
    and the original matrix reconstructed via
    unique_matrix[inverse]
    """
    lil_matrix = matrix.tolil()
    rows = [x + y for x, y in zip(lil_matrix.rows, lil_matrix.data)]
    return_values = return_counts + return_inverse + return_index
    return np.unique(
        rows,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )[1 : (return_values + 1)]


def disconnected_vertices(model):
    """
    Returns a boolean vector indicating which vertices are disconnected from the umap graph.
    These vertices will often be scattered across the space and make it difficult to focus on the main
    manifold.  They can either be filtered and have UMAP re-run or simply filtered from the interactive plotting tool
    via the subset_points parameter.
    Use ~disconnected_vertices(model) to only plot the connected points.
    Parameters
    ----------
    model: a trained UMAP model

    Returns
    -------
    A boolean vector indicating which points are disconnected
    """
    check_is_fitted(model, "graph_")
    if model.unique:
        vertices_disconnected = (
            np.array(model.graph_[model._unique_inverse_].sum(axis=1)).flatten() == 0
        )
    else:
        vertices_disconnected = np.array(model.graph_.sum(axis=1)).flatten() == 0
    return vertices_disconnected


def average_nn_distance(dist_matrix):
    """Calculate the average distance to each points nearest neighbors.

    Parameters
    ----------
    dist_matrix: a csr_matrix
        A distance matrix (usually umap_model.graph_)

    Returns
    -------
    An array with the average distance to each points nearest neighbors

    """
    (row_idx, col_idx, val) = scipy.sparse.find(dist_matrix)

    # Count/sum is done per row
    count_non_zero_elems = np.bincount(row_idx)
    sum_non_zero_elems = np.bincount(row_idx, weights=val)
    averages = sum_non_zero_elems / count_non_zero_elems

    if any(np.isnan(averages)):
        warn(
            "Embedding contains disconnected vertices which will be ignored."
            "Use umap.utils.disconnected_vertices() to identify them."
        )

    return averages

################# ADDITIONS [ejk]

@numba.njit()
def round3(x):
    """ helper for debug printing in reposition_jit. """
    return np.around(x,3)

@numba.njit()
def reposition_jit(x: numba.float32[:,:],
                   #umapper, # Cannot determine numba type of ... .UMAP
                   #up: List(Tuple([numba.int32, numba.float32[:]])),
                   # above has numba deprecation warning, so accept
                   #       just raw ndarrays instead.
                   up_idx,  # e.g. int32[:]
                   up_pos,  # e.g. float32[:,:]
                   # csr format of connection strength graph
                   indices: numba.int32[:],
                   indptr: numba.int32[:],
                   data: numba.float32[:],

                   nbr: numba.int64 = 1,            # 0: debug (move just up_idx) 1: move n.n.
                   thresh: numba.float32 = 0.0,     # use all available n.n. in umap.graph_
                   verbose: numba.int32 = 0,        # 0,1,2,3 used
                   nnfactor: numba.float32 = 1.0    # 1.0 ~ n.n. potentially "as far" up_idx pts
                   ):
    """ Move up_idx[:] points in x to positions up_pos[:,:] updates, and also [nbr==1]
    move all neighbors of up_idx[:] by some downweighted proportion.

    Base movement strength is a symmetric matrix, provided as CSR format arrays,
    possibly originating from a umapper.graph_ membership strengths.  Such strengths
    are based on the hi-D distance, rather than actual lo-D distance.

    All non-up_idx[:] neighbors move vectors is discouted individually by the strength
    matrix, with a value in [0.0,1.0].  Neighbor move distance are uniformly reduced
    by another nnfactor.

    Neighbors of multiple up_idx[:] move points are moved by the strength-weighted average
    of the individual move vectors (and also receive the additional nnfactor reduction).

    This version for clarity and debug, rather than speed!

    Parameters
    ----------
    x           Data array[npts,dim]
    up_idx      array[:] rows in x that move
    up_pos      array[len(up_idx), dim] where move pts go to

    indices, indptr, data   CSR (compressed row) sparse matrix data describing a
                            symmetric edge STRENGTH matrix with values in [0.0,1.0].
                            1.0 is strongest-connected (i.e. closest neighbor)
    nbr         1 to enable n.n. movement
    thresh      only use connection STRENGTHs greater than this
    verbose     0,1,2,3 to see what happened
    nnfactor    uniform movement reduction applied to all neighbors-of-up_idx[]

    Return
    ------
    a new low-D embedding based on x[:,:] and its explicit updates 'up_*'

    If x is a current UMAP lo-D embedding, the return value might be a good
    starting point to continue (modify) an existing umap embedding.

    Particular for lo-D == 2, neighbors can easily get "stuck" behind other points.
    The returned embedding can "spread out" a cluster based on just a few points.
    
    NOTES/TODO:
        - option to adjust thresh so NO neighborhoods intersect, so points
          move either one way or the other and never in some average direction
        - thresh==0.0 and nnfactor==1.0 might not be good defaults.
        - Also umapper.n_neighbors roughly influences how many points comprise
          each neighborhood, so again might influence a good value for thresh.
        - given the expanded movements, "alignment" with previous embedding becomes
          less meaningful.
    """
    #if thresh is None:
    #    thresh = 0.0
    # a nbr=0 trivial update (just the specified movements, no more)
    if verbose>=2: print("type(nbr)",type(nbr),"nbr",nbr,"thresh",thresh,"verbose",verbose)
    if nbr==0:
        y = x.copy()      # our return value - for no updates, exact copy
        dim = x.shape[1]  # assume sane dimensions, for now
        for p in range(up_idx.shape[0]):
            pt = up_idx[p]  # point number
            vec = up_pos[p] # new posn
            for d in range(dim):
                y[pt,d] = vec[d]
    elif nbr==1:
        if verbose>=3: print("nbr==1")
        #y = x.copy()      # our return value - for no updates, exact copy
        # each point first accumulates (sum_weights, mvmt), ... information
        dim = x.shape[1]  # assume sane dimensions, for now
        sum_wt = np.zeros(x.shape[0],dtype=np.float32)
        y = np.zeros(x.shape, dtype=x.dtype)
        # y first accumulates over moved points 'p'
        # the weighted movement vectors for all 'other' in nbrhoods of 'p'
        if verbose>=3: print("loop over pts in up_idx =",up_idx)
        for p in range(up_idx.shape[0]):
            pt = up_idx[p]          # point number
            old_pt_pos = x[pt,:]
            new_pt_pos = up_pos[p]     # new posn of pt
            # explicit instruction moves pt : vector pt old-->new
            pt_move = (new_pt_pos - old_pt_pos)
            if verbose>=1: print("pt ",pt,"old_pt_pos",old_pt_pos,"new_pt_pos",new_pt_pos,"    pt_move",pt_move,"dist",round3(norm(pt_move)))
            # record the explict movement instruction:
            for d in range(dim):
                y[pt,d] = pt_move[d]
            lo = indptr[pt]         # row pt indices start
            hi = indptr[pt+1]       # row pt indices end
            cols = indices[lo:hi]   # columns (neighbors) of this row (pt)
            vals = data[lo:hi]      # values in above columns (umap membership strengths)
            for j in range(cols.shape[0]):  # for echo 'other' colum
                other = cols[j]             # other point number
                other_pos = x[other]        # other point position
                strength = vals[j]          # umap strength[pt<->other] in [0,1]
                if strength <= thresh:       # skip if other insufficiently connected
                    continue
                if other in up_idx:         # if other is explicitly moved, don't screw with it
                    continue
                if verbose>=3: print("pt ",pt,"<--> other",other,"@",other_pos,"strength",vals[j])
                # other moves uniformly downweighted by additional nnfactor
                other_delta = nnfactor * strength * pt_move # move vector contribution of other
                if sum_wt[other] == 0.0:
                    if verbose>=3: print("   single   contrib other",other,"delta",other_delta,"dist",round3(norm(other_delta)),"strength",strength)
                else:
                    if verbose>=2: print("   multiple contrib other",other,"delta",other_delta,"dist",round3(norm(other_delta)),"strength",strength)
                sum_wt[other] += strength           # accum other vecs for different pt-nbrhoods
                y[other,:] += other_delta           # y[other] ~ vector movement sum

        # then we form the weighted-sum of movement deltas (sum over all nbrhoods touching 'p')
        for other in range(x.shape[0]):
            if sum_wt[other] != 0.0:        # sum_wt 0 ~ other is in NO nbrhood of a 'p' have sum_wt==0.0
                y[other,:] /= (sum_wt[other] + 1e-4)
                if verbose>=3: print("avg other",other,"movement",y[other,:],"dist",round3(norm(y[other,:])))
        # record the explicit delta-movements
        # (more robust to do it "again", at least for now, until error-checking done)
        for p in range(up_idx.shape[0]):
            pt = up_idx[p]
            y[pt,:] = up_pos[p] - x[pt,:]
            sum_wt[pt] = 1.0
            if verbose>=3: print("explicit update pt",pt,"by",y[pt,:],"dist",round3(norm(y[pt,:])))

        # then we apply the movement deltas, now across entire dataset
        for other in range(x.shape[0]):
            # other original pos x[other,:] moves by y[other,:] ~ weighted-average-other_delta
            y_final = x[other,:] + y[other,:]
            if sum_wt[other] > 0.0:
                if verbose>=1:
                    ptmsg = "   other" if other not in up_idx else "      pt"
                    print(ptmsg,other,"@",y[other,:],"-->",y_final,"dist",round3(norm(y[other,:])),"sum_wt",round3(sum_wt[other]))
            else:
                assert np.all(y[other,:] == 0.0)
            y[other,:] = y_final

    else:
        raise Exception("nbrhood diffusion nbr==0 or nbr==1 only")

    return y

def reposition(x, up_pts, up_pos, umapper, thresh=0.0, verbose=0, nnfactor=1.0):
    """ python wrapper for the reposition_jit.

    Arguments  (arrays are numpy arrays)
    ---------

    x       data array[pts,dim] ~ a lo-D embedding
    up_pts  int vector which pts (rows of x) explicitly move?
    up_pos  array[len(up_pts), dim] ~ final destination of those points.
    umapper umapper.graph_ is a csr-format compressed array of membership strengths in [0,1]
    thresh  operate on n.n. whose strengths are >= thresh in [0,1]
    verbose 0,1,2,3 effective

    at high thresh, up_pts neighborhoods are more likely disjoint
    at default, up_pts may have 'other' nbrs that accumulate a weighted
                average of up_pts movement deltas.  weights are taken
                directly from the umapper.graph_.
                Weighted average movement deltas are multiplied by some
                nnfactor [default 1.0]
    """
    g = umapper.graph_
    # hardwire to 1-neighborhood diffusion
    nbr = 1
    return reposition_jit(x, up_pts, up_pos, g.indices, g.indptr, g.data,
                          nbr, thresh, verbose, nnfactor )

