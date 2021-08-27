import numpy as np
import numba
import umap.distances as dist
from umap.utils import tau_rand_int
import umap.constraints as con

layout_version = 3
#from umap.constraints import DimLohi, constrain_lo, constrain_hi
#from umap.constraints import HardPinIndexed, PinNoninf


@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit()
def clip_array(arr):
    """ clip array elementwise. """
    return np.where( arr > 4.0, 4.0,
                    np.where(arr < -4.0, -4.0, arr))

@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

# Workaround:
# compose tuple of jitted fns(idx,pt)
# This avoid numba 0.53 "experimental" passing functions warnings.
#
# When numba 0.53 is required, this hackery is not required
# ref: https://github.com/numba/numba/issues/3405  and older 2542
# 
@numba.njit()
def _fn_idx_pt_noop(idx,pt):
    return None
def _chain_idx_pt(fs, inner=None):
    """for f(idx,pt) in tuple fs, invoke them in order"""
    if len(fs) == 0:
        assert inner is None
        return _fn_idx_pt_noop
    head = fs[0]
    if inner is None:
        @numba.njit()
        def wrap(idx,pt):
            head(idx,pt)
        return wrap
    else:
        @numba.njit()
        def wrap(idx,pt):
            inner(idx,pt)
            head(idx,pt)
    if len(fs) > 1:
        tail = fs[1:]
        return _chain_idx_pt(tail, wrap)
    else:
        return wrap
# Now to chain jit fns that don't need an index
# Ex. numba wrappers for dimlohi_pt(pt, los, his) in constraints.py
@numba.njit()
def _fn_pt_noop(pt):
    return None
def _chain_pt(fs, inner=None):
    """for f(idx,pt) in tuple fs, invoke them in order"""
    if len(fs) == 0:
        assert inner is None
        return _fn_pt_noop
    head = fs[0]
    if inner is None:
        @numba.njit()
        def wrap(pt):
            head(pt)
    else:
        @numba.njit()
        def wrap(pt):
            inner(pt)
            head(pt)
    if len(fs) > 1:
        tail = fs[1:]
        return _chain_pt(tail, wrap)
    else:
        return wrap
#


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,              # epoch
    wrap_idx_pt,
    wrap_idx_grad,
    densmap_flag,
    dens_phi_sum,
    dens_re_sum,
    dens_re_cov,
    dens_re_std,
    dens_re_mean,
    dens_lambda,
    dens_R,
    dens_mu,
    dens_mu_tot,
):
    #if constraints is None:
    #    constraints = {}

    #if 'grad' in constraints:
    #    # grad constraints include hard-pinning.  This needs points to be "OK"
    #    # before zeroing the gradients in tangent_space.   Soft force constraints
    #    # probably have these as no-ops.
    #    for constraint in constraints['grad']:
    #        # UNTESTED
    #        constraint.project_rows_onto_constraint(head_embedding)
    #        constraint.project_rows_onto_constraint(tail_embedding)
    #        # once constrained, iteration can just do project_onto_tangent_space
    #        #                   to zero the required gradients.

    #if len(constrain_idx_pt):
    #print("_optimize_layout_euclidean_single_epoch")
    #print("head,tail shapes",head_embedding.shape, tail_embedding.shape)
    #for j in [13,14]:
    #    print("init head_embeding[",j,"]", head_embedding[j])
    #    print("init tail_embeding[",j,"]", tail_embedding[j])
    assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    assert np.all(head_embedding == tail_embedding)

    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]
            #if len(constrain_idx_pt) and j==13: print("+head13", head_embedding[13])
            #if len(constrain_idx_pt) and k==13: print("+tail13", head_embedding[13])

            dist_squared = rdist(current, other)

            if densmap_flag:
                phi = 1.0 / (1.0 + a * pow(dist_squared, b))
                dphi_term = (
                    a * b * pow(dist_squared, b - 1) / (1.0 + a * pow(dist_squared, b))
                )

                q_jk = phi / dens_phi_sum[k]
                q_kj = phi / dens_phi_sum[j]

                drk = q_jk * (
                    (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[k]) + dphi_term
                )
                drj = q_kj * (
                    (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[j]) + dphi_term
                )

                re_std_sq = dens_re_std * dens_re_std
                weight_k = (
                    dens_R[k]
                    - dens_re_cov * (dens_re_sum[k] - dens_re_mean) / re_std_sq
                )
                weight_j = (
                    dens_R[j]
                    - dens_re_cov * (dens_re_sum[j] - dens_re_mean) / re_std_sq
                )

                grad_cor_coeff = (
                    dens_lambda
                    * dens_mu_tot
                    * (weight_k * drk + weight_j * drj)
                    / (dens_mu[i] * dens_re_std)
                    / n_vertices
                )

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            # original attractivie-update loop
            #for d in range(dim):
            #    grad_d = clip(grad_coeff * (current[d] - other[d]))
            #
            #    if densmap_flag:
            #        grad_d += clip(2 * grad_cor_coeff * (current[d] - other[d]))
            #
            #    current[d] += grad_d * alpha
            #    if move_other:
            #        other[d] += -grad_d * alpha
            #
            # replacement: grad_d vector might be projected onto tangent space
            delta = current - other # vector[dim]
            grad_d = clip_array(grad_coeff * delta)
            if densmap_flag:
                grad_d += clip_array(2 * grad_cor_coeff * delta)
            current_grad = grad_d.copy()
            if wrap_idx_grad is not None:
                wrap_idx_grad(j, current, current_grad) # modify current_grad
                current_grad = clip_array(current_grad)
            current += alpha * current_grad
            if wrap_idx_pt is not None:
                wrap_idx_pt(j, current)

            if move_other:
                other_grad = -grad_d
                #if grad_constraint is not None:
                #    grad_constraint.project_index_onto_tangent_space(k, other, grad_d)
                if wrap_idx_grad is not None:
                    wrap_idx_grad(k, other, other_grad) # modify other_grad
                    other_grad = clip_array(other_grad)
                other += alpha * other_grad
                if wrap_idx_pt is not None:
                    wrap_idx_pt(k, other)

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                if grad_coeff > 0.0:
                    grad_d = clip_array(grad_coeff * (current - other))
                else:
                    # [ejk] quite strange.
                    #       Perhaps "do anything to avoid accidental superpositions"
                    grad_d = np.full(dim, 4.0)
                current_grad = grad_d.copy()
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, current_grad) # modify current_grad
                    current_grad = clip_array(current_grad)
                current += alpha * current_grad
                if True and wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)

                # following is needed for correctness if tail==head
                if move_other:
                    other_grad = -grad_d
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(k, other, other_grad) # modify other_grad
                        other_grad = clip_array(other_grad)
                    other += alpha * other_grad
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(k, other)

            #if len(constrain_idx_pt) and j==13: print("+head13", head_embedding[13])
            #if len(constrain_idx_pt) and k==13: print("+tail13", head_embedding[13])

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
    #if len(constrain_idx_pt):
    #print("final head_embeding[13]", head_embedding[13])
    #print("final tail_embeding[13]", tail_embedding[13])


def _optimize_layout_euclidean_densmap_epoch_init(
    head_embedding,
    tail_embedding,
    head,
    tail,
    a,
    b,
    re_sum,
    phi_sum,
):
    re_sum.fill(0)
    phi_sum.fill(0)

    for i in numba.prange(head.size):
        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]
        dist_squared = rdist(current, other)

        phi = 1.0 / (1.0 + a * pow(dist_squared, b))

        re_sum[j] += phi * dist_squared
        re_sum[k] += phi * dist_squared
        phi_sum[j] += phi
        phi_sum[k] += phi

    epsilon = 1e-8
    for i in range(re_sum.size):
        re_sum[i] = np.log(epsilon + (re_sum[i] / phi_sum[i]))


def optimize_layout_euclidean_nomask(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    densmap=False,
    densmap_kwds={},
    move_other=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    densmap: bool (optional, default False)
        Whether to use the density-augmented densMAP objective
    densmap_kwds: dict (optional, default {})
        Auxiliary data for densMAP
    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    if True:
        # TRIAL: constrain 'x' (first) embedding dim to [-1,+1] range
        # (we can safely specify fewer (the first) dims to be bounded)
        #los = np.array([constrain_lo], dtype=np.float32)
        #his = np.array([constrain_hi], dtype=np.float32)
        #constraints = {
        #    "point": [DimLohi(los, his)]
        #}
        #point_constraint = DimLohi(los,his)

        return optimize_layout_euclidean_masked(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma=1.0,
            initial_alpha=1.0,
            negative_sample_rate=5.0,
            parallel=False,
            verbose=False,
            densmap=False,
            densmap_kwds={},
            move_other=False,
            pin_mask=None,
        )

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    #if pin_mask is not None:
    #    assert (pin_mask.shape == head_embedding.shape
    #            or pin_mask.shape == torch.Size(head_embedding.shape[0]))

    optimize_fn = numba.njit(
        _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel
    )

    if densmap:
        dens_init_fn = numba.njit(
            _optimize_layout_euclidean_densmap_epoch_init,
            fastmath=True,
            parallel=parallel,
        )

        dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2
        dens_lambda = densmap_kwds["lambda"]
        dens_R = densmap_kwds["R"]
        dens_mu = densmap_kwds["mu"]
        dens_phi_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_re_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_var_shift = densmap_kwds["var_shift"]
    else:
        dens_mu_tot = 0
        dens_lambda = 0
        dens_R = np.zeros(1, dtype=np.float32)
        dens_mu = np.zeros(1, dtype=np.float32)
        dens_phi_sum = np.zeros(1, dtype=np.float32)
        dens_re_sum = np.zeros(1, dtype=np.float32)


    for n in range(n_epochs):

        densmap_flag = (
            densmap
            and (densmap_kwds["lambda"] > 0)
            and (((n + 1) / float(n_epochs)) > (1 - densmap_kwds["frac"]))
        )

        if densmap_flag:
            dens_init_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                a,
                b,
                dens_re_sum,
                dens_phi_sum,
            )

            dens_re_std = np.sqrt(np.var(dens_re_sum) + dens_var_shift)
            dens_re_mean = np.mean(dens_re_sum)
            dens_re_cov = np.dot(dens_re_sum, dens_R) / (n_vertices - 1)
        else:
            dens_re_std = 0
            dens_re_mean = 0
            dens_re_cov = 0

        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
            # NO: pin_mask,
            densmap_flag,
            dens_phi_sum,
            dens_re_sum,
            dens_re_cov,
            dens_re_std,
            dens_re_mean,
            dens_lambda,
            dens_R,
            dens_mu,
            dens_mu_tot,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    # epoch constraints:
    #      retain all relative distances of embedding.
    # i.e. they are some translate, rotate and uniform scaling
    #      Ex. cemter of mass at origin, and rms radius = 1
    if 'epoch' in constraints:
        for constraint in constraints['epoch']:
            constraint.project_onto_constraint(head_embedding)

    return head_embedding


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    densmap=False,
    densmap_kwds={},
    move_other=False,
    output_constrain=None, # TBD: independent of head_embedding index
    pin_mask=None, # dict of constraints (or ndarray of 0/1 mask)
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    output_constrain: default None
        dict of numba point or grad submanifold projection functions
    pin_mask: default None
        Array of shape (n_samples) or (n_samples, n_components)
        The weights (in [0,1]) assigned to each sample, defining how much they
        should be updated. 0 means the point will not move at all, 1 means
        they are updated normally. In-between values allow for fine-tuning.
        A 2-D mask can supply different weights to each dimension of each sample.
        OR
        A dictionary of strings -> [constraint,...] (WIP)
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    densmap: bool (optional, default False)
        Whether to use the density-augmented densMAP objective
    densmap_kwds: dict (optional, default {})
        Auxiliary data for densMAP
    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    print("optimize_layout_euclidean_masked")
    print("head,tail shapes",head_embedding.shape, tail_embedding.shape)
    dim = head_embedding.shape[1]
    #move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    if pin_mask is None:
        pin_mask = {}
    have_constraints = True
    # historical: packing a list of fns into a single call was problematic
    #             perhaps better in numba >= 0.53?
    # But eventually it would be nice to accept a list of functions
    fns_idx_pt = []
    fns_idx_grad = []

    if isinstance(pin_mask, dict): # pin_mask is a more generic "constraints" dictionary
        # Dictionary layout:
        #   key  -->   list( tuple )
        #   where tuple ~ (jit_function, [const_args,...]) (or empty)
        #print("constraints", pin_mask.keys())
        for kk,k in enumerate(pin_mask.keys()):
            print("kk,k",kk,k)
            if k=='idx_pt':
                fns_idx_pt = [pin_mask[k]] # revert: do NOT support a list, numba issues?
            elif k=='idx_grad':
                fns_idx_grad = [pin_mask[k]]
            #elif k=='epoch': fns_epoch = [pin_mask[k][0]]
            else:
                print(" Warning: unrecognized constraints key", k)
                print(" Allowed constraint keys:", recognized)
        # I'm trying to avoid numba 0.53 warnings (about firstclass functions)

        optimize_fn = numba.njit(
            _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel,
        )
    else:
        have_constraints = False
        print("pin_mask")
        assert (isinstance(pin_mask, np.ndarray))
        assert (pin_mask.shape == head_embedding.shape
                or pin_mask.shape == torch.Size(head_embedding.shape[0]))
        # DEBUG:
        for i in range(pin_mask.shape[0]):
            for d in range(dim):
                if pin_mask[i,d] == 0.0:
                    print("sample",i,"pin head[",d,"] begins at",head_embedding[i,d])

        # v.2 translate zeros in pin_mask to embedding values; nonzeros become np.inf
        # and then uses a 'freeinf' constraint from constraints.py
        have_constraints = True
        freeinf_arg = np.where( pin_mask == 0.0, head_embedding, np.float32(np.inf) )
        # original approach
        @numba.njit()
        def pin_mask_constraint(idx,pt):
            con.freeinf_pt( idx, pt, freeinf_arg )
        fns_idx_pt = [pin_mask_constraint,]
        # OR mirror a tuple-based approach
        #pin_mask_constraint_tuple = (con.freeinf_pt, freeinf_arg,)
        #@numba.njit()
        #def pin_mask_constraint2(idx,pt):
        #    pin_mask_constraint_tuple[0]( idx, pt, *pin_mask_constraint_tuple[1:] )
        #fns_idx_pt += (pin_mask_constraint2,)

        optimize_fn = numba.njit(
            _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel,
        )

    # call a tuple of jit functions(idx,pt) in sequential order
    print("fns_idx_pt",fns_idx_pt)
    # Note: _chain_idx_pt has some numba issues
    #       todo: get rid of list fns_idx_pt etc (not useful)
    wrap_idx_pt = None # or con.noop_pt
    wrap_idx_grad = None # or con.noop_grad
    if len(fns_idx_pt):
        #wrap_idx_pt = _chain_idx_pt( fns_idx_pt )
        if len(fns_idx_pt) > 1:
            print(" Warning: only accepting 1st idx_pt constraint for now")
        wrap_idx_pt = fns_idx_pt[0]
    if len(fns_idx_grad):
        #wrap_idx_grad = _chain_idx_grad( fns_idx_pt )
        if len(fns_idx_grad) > 1:
            print(" Warning: only accepting 1st idx_grad constraint for now")
        wrap_idx_grad = fns_idx_grad[0]

    if densmap:
        dens_init_fn = numba.njit(
            _optimize_layout_euclidean_densmap_epoch_init,
            fastmath=True,
            parallel=parallel,
        )

        dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2
        dens_lambda = densmap_kwds["lambda"]
        dens_R = densmap_kwds["R"]
        dens_mu = densmap_kwds["mu"]
        dens_phi_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_re_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_var_shift = densmap_kwds["var_shift"]
    else:
        dens_mu_tot = 0
        dens_lambda = 0
        dens_R = np.zeros(1, dtype=np.float32)
        dens_mu = np.zeros(1, dtype=np.float32)
        dens_phi_sum = np.zeros(1, dtype=np.float32)
        dens_re_sum = np.zeros(1, dtype=np.float32)

    for n in range(n_epochs):

        densmap_flag = (
            densmap
            and (densmap_kwds["lambda"] > 0)
            and (((n + 1) / float(n_epochs)) > (1 - densmap_kwds["frac"]))
        )

        if densmap_flag:
            dens_init_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                a,
                b,
                dens_re_sum,
                dens_phi_sum,
            )

            dens_re_std = np.sqrt(np.var(dens_re_sum) + dens_var_shift)
            dens_re_mean = np.mean(dens_re_sum)
            dens_re_cov = np.dot(dens_re_sum, dens_R) / (n_vertices - 1)
        else:
            dens_re_std = 0
            dens_re_mean = 0
            dens_re_cov = 0

        if have_constraints:

            optimize_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                rng_state,
                gamma,
                dim,
                move_other,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                wrap_idx_pt,
                wrap_idx_grad,
                densmap_flag,
                dens_phi_sum,
                dens_re_sum,
                dens_re_cov,
                dens_re_std,
                dens_re_mean,
                dens_lambda,
                dens_R,
                dens_mu,
                dens_mu_tot,
            )

        else:
            optimize_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                pin_mask,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                rng_state,
                gamma,
                dim,
                move_other,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                densmap_flag,
                dens_phi_sum,
                dens_re_sum,
                dens_re_cov,
                dens_re_std,
                dens_re_mean,
                dens_lambda,
                dens_R,
                dens_mu,
                dens_mu_tot,
            )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding


@numba.njit(fastmath=True)
def optimize_layout_generic(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=dist.euclidean,
    output_metric_kwds=(),
    verbose=False,
    move_other=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    weight: array of shape (n_1_simplices)
        The membership weights of the 1-simplices.

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )
                _, rev_grad_dist_output = output_metric(
                    other, current, *output_metric_kwds
                )

                if dist_output > 0.0:
                    w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                else:
                    w_l = 1.0
                grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])

                    current[d] += grad_d * alpha
                    if move_other:
                        grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                        other[d] += grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_output, grad_dist_output = output_metric(
                        current, other, *output_metric_kwds
                    )

                    if dist_output > 0.0:
                        w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                    elif j == k:
                        continue
                    else:
                        w_l = 1.0

                    grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                    for d in range(dim):
                        grad_d = clip(grad_coeff * grad_dist_output[d])
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding


@numba.njit(fastmath=True)
def optimize_layout_inverse(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    sigmas,
    rhos,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=dist.euclidean,
    output_metric_kwds=(),
    verbose=False,
    move_other=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    weight: array of shape (n_1_simplices)
        The membership weights of the 1-simplices.

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )

                w_l = weight[i]
                grad_coeff = -(1 / (w_l * sigmas[k] + 1e-6))

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])

                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_output, grad_dist_output = output_metric(
                        current, other, *output_metric_kwds
                    )

                    # w_l = 0.0 # for negative samples, the edge does not exist
                    w_h = np.exp(-max(dist_output - rhos[k], 1e-6) / (sigmas[k] + 1e-6))
                    grad_coeff = -gamma * ((0 - w_h) / ((1 - w_h) * sigmas[k] + 1e-6))

                    for d in range(dim):
                        grad_d = clip(grad_coeff * grad_dist_output[d])
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding


def _optimize_layout_aligned_euclidean_single_epoch(
    head_embeddings,
    tail_embeddings,
    heads,
    tails,
    epochs_per_sample,
    a,
    b,
    regularisation_weights,
    relations,
    rng_state,
    gamma,
    lambda_,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    n_embeddings = len(heads)
    window_size = (relations.shape[1] - 1) // 2

    max_n_edges = 0
    for e_p_s in epochs_per_sample:
        if e_p_s.shape[0] >= max_n_edges:
            max_n_edges = e_p_s.shape[0]

    embedding_order = np.arange(n_embeddings).astype(np.int32)
    np.random.seed(abs(rng_state[0]))
    np.random.shuffle(embedding_order)

    for i in range(max_n_edges):
        for m in embedding_order:
            if i < epoch_of_next_sample[m].shape[0] and epoch_of_next_sample[m][i] <= n:
                j = heads[m][i]
                k = tails[m][i]

                current = head_embeddings[m][j]
                other = tail_embeddings[m][k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))

                    for offset in range(-window_size, window_size):
                        neighbor_m = m + offset
                        if (
                            neighbor_m >= 0
                            and neighbor_m < n_embeddings
                            and offset != 0
                        ):
                            identified_index = relations[m, offset + window_size, j]
                            if identified_index >= 0:
                                grad_d -= clip(
                                    (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                    * regularisation_weights[m, offset + window_size, j]
                                    * (
                                        current[d]
                                        - head_embeddings[neighbor_m][
                                            identified_index, d
                                        ]
                                    )
                                )

                    current[d] += clip(grad_d) * alpha
                    if move_other:
                        other_grad_d = clip(grad_coeff * (other[d] - current[d]))

                        for offset in range(-window_size, window_size):
                            neighbor_m = m + offset
                            if (
                                neighbor_m >= 0
                                and neighbor_m < n_embeddings
                                and offset != 0
                            ):
                                identified_index = relations[m, offset + window_size, k]
                                if identified_index >= 0:
                                    grad_d -= clip(
                                        (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                        * regularisation_weights[
                                            m, offset + window_size, k
                                        ]
                                        * (
                                            other[d]
                                            - head_embeddings[neighbor_m][
                                                identified_index, d
                                            ]
                                        )
                                    )

                        other[d] += clip(other_grad_d) * alpha

                epoch_of_next_sample[m][i] += epochs_per_sample[m][i]

                if epochs_per_negative_sample[m][i] > 0:
                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[m][i])
                        / epochs_per_negative_sample[m][i]
                    )
                else:
                    n_neg_samples = 0

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % tail_embeddings[m].shape[0]

                    other = tail_embeddings[m][k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0

                        for offset in range(-window_size, window_size):
                            neighbor_m = m + offset
                            if (
                                neighbor_m >= 0
                                and neighbor_m < n_embeddings
                                and offset != 0
                            ):
                                identified_index = relations[m, offset + window_size, j]
                                if identified_index >= 0:
                                    grad_d -= clip(
                                        (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                        * regularisation_weights[
                                            m, offset + window_size, j
                                        ]
                                        * (
                                            current[d]
                                            - head_embeddings[neighbor_m][
                                                identified_index, d
                                            ]
                                        )
                                    )

                        current[d] += clip(grad_d) * alpha

                epoch_of_next_negative_sample[m][i] += (
                    n_neg_samples * epochs_per_negative_sample[m][i]
                )


def optimize_layout_aligned_euclidean(
    head_embeddings,
    tail_embeddings,
    heads,
    tails,
    n_epochs,
    epochs_per_sample,
    regularisation_weights,
    relations,
    rng_state,
    a=1.576943460405378,
    b=0.8950608781227859,
    gamma=1.0,
    lambda_=5e-3,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=True,
    verbose=False,
    move_other=False,
):
    dim = head_embeddings[0].shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = numba.typed.List.empty_list(numba.types.float32[::1])
    epoch_of_next_negative_sample = numba.typed.List.empty_list(
        numba.types.float32[::1]
    )
    epoch_of_next_sample = numba.typed.List.empty_list(numba.types.float32[::1])

    for m in range(len(heads)):
        epochs_per_negative_sample.append(
            epochs_per_sample[m].astype(np.float32) / negative_sample_rate
        )
        epoch_of_next_negative_sample.append(
            epochs_per_negative_sample[m].astype(np.float32)
        )
        epoch_of_next_sample.append(epochs_per_sample[m].astype(np.float32))

    optimize_fn = numba.njit(
        _optimize_layout_aligned_euclidean_single_epoch,
        fastmath=True,
        parallel=parallel,
    )

    for n in range(n_epochs):
        optimize_fn(
            head_embeddings,
            tail_embeddings,
            heads,
            tails,
            epochs_per_sample,
            a,
            b,
            regularisation_weights,
            relations,
            rng_state,
            gamma,
            lambda_,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embeddings
#
