import numpy as np
import numba
import umap.distances as dist
from umap.utils import tau_rand_int
import umap.constraints as con

layout_version = 3
#from umap.constraints import DimLohi, constrain_lo, constrain_hi
#from umap.constraints import HardPinIndexed, PinNoninf
#from numba import types


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
    """ clip array elementwise.  This is MUCH SLOWER than using scalar clip(val) ! """
    return np.where( arr > 4.0, 4.0,
                    np.where(arr < -4.0, -4.0, arr))
    #for d in arr.shape[0]:
    #    arr[d] = 4.0 if arr[d] > 4.0 else -4.0 if arr[d] < -4.0 else arr[d]

@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    inline='always',
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "d": numba.types.int32,
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
    for d in range(dim):
        diff = x[d] - y[d]
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

#
# REMOVED apply_grad -- it is MUCH faster when inlined
#                     -- it is only used 4 times
# for code clarity ... reduce code duplication
# Is numba jit able to elide "is not None" code blocks?
#    Quick tests show that using 'None' (or optional), numba may fail
#    to elide the code block :(  (used env NUMBA_DEBUG to check simple cases)
#@numba.njit() # use to inspect
@numba.njit(inline='always')
def apply_grad(idx, pt, alpha, grad, fn_idx_grad, fn_grad, fn_idx_pt, fn_pt):
    """ updates pt by (projected?) grad and any pt projection functions.

    idx:    point number
    pt:     point vector
    alpha:  learning rate
    grad:   gradient vector
    fn_...: constraint functions (or None)
    
    Both pt and grad may be modified.
    """
    # can i get _doclip as a constant?
    _doclip = False
    #_doclip = (fn_idx_grad is not None or fn_grad is not None)
    #_doclip = isinstance(fn_idx_grad, types.NoneType) or isinstance(fn_grad,  types.NoneType)
    if fn_idx_grad is not None:
        fn_idx_grad(idx, pt, grad)  # grad (pt?) may change
        _doclip = True
    if fn_grad is not None:
        fn_grad(pt, grad)         # grad (pt?) may change
        _doclip = True
    if _doclip:
        for d in range(pt.shape[0]):
            pt[d] = pt[d] + alpha * clip(grad[d])
    else:
        for d in range(pt.shape[0]):
            pt[d] = pt[d] + alpha * grad[d]

    if fn_idx_pt is not None:
        fn_idx_pt(idx, pt)
    if fn_pt is not None:
        fn_pt(pt)
    # no return value -- pt and grad mods are IN-PLACE.
@numba.njit() # use to inspect
#@numba.njit(inline='always')
def apply_grad0(idx, pt, alpha, grad, fn_idx_grad, fn_grad, fn_idx_pt, fn_pt):
    """ updates pt by (projected?) grad and any pt projection functions.

    idx:    point number
    pt:     point vector
    alpha:  learning rate
    grad:   gradient vector
    fn_...: constraint functions (or None)
    
    Both pt and grad may be modified.
    """
    # can i get _doclip as a constant?
    _doclip = (fn_idx_grad is not None or fn_grad is not None)
    if fn_idx_grad is not None:
        fn_idx_grad(idx, pt, grad)  # grad (pt?) may change
        #_doclip = True
    if fn_grad is not None:
        fn_grad(pt, grad)         # grad (pt?) may change
        #_doclip = True
    if _doclip:
        for d in range(pt.shape[0]):
            pt[d] = pt[d] + alpha * clip(grad[d])
    else:
        for d in range(pt.shape[0]):
            pt[d] = pt[d] + alpha * grad[d]

    if fn_idx_pt is not None:
        fn_idx_pt(idx, pt)
    if fn_pt is not None:
        fn_pt(pt)
    # no return value -- pt and grad mods are IN-PLACE.

# did not work out...
#@numba.generated_jit(nopython=True)
#def apply_grad2(idx, pt, alpha, grad, fn_idx_grad, fn_grad, fn_idx_pt, fn_pt):
#    """ updates pt by (projected?) grad and any pt projection functions.
#
#    idx:    point number
#    pt:     point vector
#    alpha:  learning rate
#    grad:   gradient vector
#    fn_...: constraint functions (or None)
#    
#    Both pt and grad may be modified.
#    """
#    prog = []
#    _doclip = False
#    if not isinstance(fn_idx_grad,type(None)):
#        prog.append("fn_idx_grad(idx, pt, grad)") # grad (pt?) may change
#        _doclip = True
#    if not isinstance(fn_grad,type(None)):
#        prog.append("fn_grad(pt, grad)")          # grad (pt?) may change
#        _doclip = True
#    if _doclip:
#        # inplace clipping fn ?
#        prog.append("grad = clip_array(grad)")
#
#    prog.append("pt += alpha * grad")
#
#    if not isinstance(fn_idx_pt,type(None)):
#        prog.append("fn_idx_pt(idx, pt)")
#    if not isinstance(fn_pt,type(None)):
#        prog.append("fn_pt(pt)")
#    
#    foostr = 'def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):\n' + '\n'.join(' '+l for l in prog)
#    print(foostr)
#    eval(foostr)
#    return myfoo

#@numba.generated_jit(nopython=True)
##@numba.extending.overload(apply_grad)
#def apply_grad3(idx, pt, alpha, grad, fn_idx_grad, fn_grad, fn_idx_pt, fn_pt):
#    """ updates pt by (projected?) grad and any pt projection functions.
#
#    idx:    point number
#    pt:     point vector
#    alpha:  learning rate
#    grad:   gradient vector
#    fn_...: constraint functions (or None)
#    
#    Both pt and grad may be modified.
#    """
#    if fn_idx_grad is not None:
#        if fn_grad is not None:
#            if fn_idx_pt is not None:
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else: # no fn_pt:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#            else: # no fn_idx_pt
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#        else: # no fn_grad
#            if fn_idx_pt is not None:
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else: # no fn_pt:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#            else: # no fn_idx_pt
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#    else: # no fn_idx_grad
#        if fn_grad is not None:
#            if fn_idx_pt is not None:
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else: # no fn_pt:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#            else: # no fn_idx_pt
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        fn_grad(pt, grad)
#                        grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#        else: # no fn_grad
#            if fn_idx_pt is not None:
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        #grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else: # no fn_pt:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        #grad = clip_array(grad)
#                        pt += alpha * grad
#                        fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#            else: # no fn_idx_pt
#                if fn_pt is not None:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        #grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        fn_pt(pt)
#                    return myfoo
#                else:
#                    def myfoo(idx,pt,alpha,grad,fn_idx_grad,fn_grad,fn_idx_pt,fn_pt):
#                        #fn_idx_grad(idx, pt, grad)
#                        #fn_grad(pt, grad)
#                        #grad = clip_array(grad)
#                        pt += alpha * grad
#                        #fn_idx_pt(idx, pt)
#                        #fn_pt(pt)
#                    return myfoo
#    raise Exception('unhandled apply_grad case')


def _optimize_layout_euclidean_single_epoch_applygrad(
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
    wrap_idx_pt,    # NEW: constraint fn(idx,pt) or None
    wrap_idx_grad,  #      constraint fn(idx,pt,grad) or None
    wrap_pt,        #      constraint fn(pt) or None
    wrap_grad,      #      constraint fn(pt,grad) or None
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

    # SUBJECT TO CHANGE ... should this be a user responsibility?
    #                       or handled in some other place?
    # WHEN CONSTRAINTS WERE FULLY-FLESHED-OUT PROJECTION OBJECTS ...
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
    #if wrap_idx_pt is not None: print("wrap_idx_pt",wrap_idx_pt)
    #if wrap_idx_grad is not None: print("wrap_idx_grad",wrap_idx_grad)
    #if wrap_pt is not None: print("wrap_pt",wrap_pt)
    #if wrap_grad is not None: print("wrap_grad",wrap_grad)

    # catch simplifying assumptions
    #   TODO: Work through constraint correctness when these fail!
    # no vis effect on speed...
    #assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    #assert np.all(head_embedding == tail_embedding)

    # mods: optimized in this order...
    # cliparray     yes: 0.263, 3.755, 0.265        no: .150, 2.08, .146    do not use clip_array
    # less np vec   yes: .146 .780 .159             no: --"--
    # more if?      yes: .151 .756 .055 (commented out)
    for i in numba.prange(epochs_per_sample.shape[0]):
        grad_d     = np.empty(dim, dtype=head_embedding.dtype)
        other_grad = np.empty(dim, dtype=head_embedding.dtype)
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if densmap_flag:
                # Note that densmap could, if you wish, be viewed as a
                # "constraint" in that it's a "gradient modification function"
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
            if False:
                delta = current - other # vector[dim]
                grad_d = clip_array(grad_coeff * delta)
                if densmap_flag:
                    grad_d += clip_array(2 * grad_cor_coeff * delta)
                other_grad = -grad_d.copy()
            else:
                if densmap_flag:
                    for d in range(dim):
                        gd = clip((grad_coeff + 2*grad_cor_coeff)
                                  * (current[d]-other[d]))
                        grad_d[d]     = gd
                        other_grad[d] = -gd
                    #if move_other:
                    #    for d in range(dim):
                    #        gd = clip((grad_coeff + 2*grad_cor_coeff)
                    #                  * (current[d]-other[d]))
                    #        grad_d[d]     = gd
                    #        other_grad[d] = -gd
                    #else:
                    #    for d in range(dim):
                    #        grad_d[d] = clip((grad_coeff + 2*grad_cor_coeff)
                    #                         * (current[d]-other[d]))
                else:
                    for d in range(dim):
                        gd = clip(grad_coeff * (current[d]-other[d]))
                        grad_d[d]     = gd
                        other_grad[d] = -gd
                    #if move_other:
                    #    for d in range(dim):
                    #        gd = clip(grad_coeff * (current[d]-other[d]))
                    #        grad_d[d]     = gd
                    #        other_grad[d] = -gd
                    #else:
                    #    for d in range(dim):
                    #        grad_d[d] = clip(grad_coeff * (current[d]-other[d]))

            apply_grad(j, current, alpha, grad_d,
                        wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)

            if move_other:
                apply_grad(k, other, alpha, other_grad,
                            wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)

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

                if False:
                    if grad_coeff > 0.0:
                        grad_d = clip_array(grad_coeff * (current - other))
                    else:
                        # [ejk] seems strange
                        #       Is this "anything to avoid accidental superpositions"?
                        #       (why not randomly +/-4?)
                        grad_d = np.full(dim, 4.0)
                    other_grad = -grad_d.copy()
                else:
                    if grad_coeff > 0.0:
                        for d in range(dim):
                            gd = clip(grad_coeff * (current[d] - other[d]))
                            grad_d[d]     = gd
                            other_grad[d] = -gd
                        #if move_other:
                        #    for d in range(dim):
                        #        gd = clip(grad_coeff * (current[d] - other[d]))
                        #        grad_d[d]     = gd
                        #        other_grad[d] = -gd
                        #else:
                        #    for d in range(dim):
                        #        grad_d[d] = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        for d in range(dim):
                            grad_d[d]     = 4.0
                            other_grad[d] = -4.0
                        #if move_other:
                        #    for d in range(dim):
                        #        grad_d[d]     = 4.0
                        #        other_grad[d] = -4.0
                        #else:
                        #    for d in range(dim):
                        #        grad_d[d]     = 4.0

                apply_grad(j, current, alpha, grad_d,
                            wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)

                # following is needed for correctness if tail==head
                if move_other:
                    apply_grad(k, other, alpha, other_grad,
                                wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
        # END if epoch_of_next_sample[i] <= n:
    # END for i in numba.prange(epochs_per_sample.shape[0]):

def _optimize_layout_euclidean_single_epoch_options(
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
    wrap_idx_pt,    # NEW: constraint fn(idx,pt) or None
    wrap_idx_grad,  #      constraint fn(idx,pt,grad) or None
    wrap_pt,        #      constraint fn(pt) or None
    wrap_grad,      #      constraint fn(pt,grad) or None
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

    # SUBJECT TO CHANGE ... should this be a user responsibility?
    #                       or handled in some other place?
    # WHEN CONSTRAINTS WERE FULLY-FLESHED-OUT PROJECTION OBJECTS ...
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
    #if wrap_idx_pt is not None: print("wrap_idx_pt",wrap_idx_pt)
    #if wrap_idx_grad is not None: print("wrap_idx_grad",wrap_idx_grad)
    #if wrap_pt is not None: print("wrap_pt",wrap_pt)
    #if wrap_grad is not None: print("wrap_grad",wrap_grad)

    # catch simplifying assumptions
    #   TODO: Work through constraint correctness when these fail!
    assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    assert np.all(head_embedding == tail_embedding)

    _doclip = wrap_idx_grad is not None or wrap_grad is not None

    #useapply_grad = 0 # use apply_grad
    #useapply_grad = 1 # v1 (longhand)
    useapply_grad = 2 # reduce # of conditionals... v2
    # 0:  .149, .780, .056 
    # 1:  .144, .735, .052
    # 2:  .151, .731, .051
    # inline always:    .147 .750 .053


    # fast, but does NOT support parallel
    # XXX TODO allocate tmp vector of size num_threads(), call avail in all 3 numba thread backends.
    grad_d     = np.empty(dim, dtype=head_embedding.dtype)
    other_grad = np.empty(dim, dtype=head_embedding.dtype)
    for i in numba.prange(epochs_per_sample.shape[0]):
        # much slower, but probably ok for parallel
        #grad_d     = np.empty(dim, dtype=head_embedding.dtype)
        #other_grad = np.empty(dim, dtype=head_embedding.dtype)
        #
        # TODO: XXX figure out how to do FAST local variables
        #
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if densmap_flag:
                #grad_cor_coeff(dist_squared, a,b,
                #               dens_phi_sum, dens_re_sum, dens_re_cov, dens_re_std,
                #               dens_re_mean, dens_lambda, dens_R, dens_mu, dens_mu_tot)
                # Note that densmap could, if you wish, be viewed as a
                # "constraint" in that it's a "gradient modification function"
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
            #delta = current - other # vector[dim]
            #grad_d = clip_array(grad_coeff * delta)
            #if densmap_flag:
            #    grad_d += clip_array(2 * grad_cor_coeff * delta)
            if densmap_flag:
                for d in range(dim):
                    delta = current[d] - other[d]
                    #grad_d[d] = clip(grad_coeff * delta)
                    #grad_d[d] += clip(2.0 * grad_cor_coeff * delta)
                    grad_d[d] = clip((grad_coeff + 2*grad_cor_coeff)
                                     * (current[d]-other[d]))
                    other_grad[d] = -grad_d[d]
            else:
                for d in range(dim):
                    grad_d[d] = clip(grad_coeff * current[d] - other[d])
                    other_grad[d] = -grad_d[d]
                #if move_other:
                #    other_grad[d] = -grad_d[d]
            # simplification (a little non-equivalent)
            # TODO: verify shape of grad_cor_coeff and do only a single clip ...


            #if move_other:
            #    other_grad = -grad_d.copy()
            #current_grad = grad_d
            #print(current.dtype, grad_d.dtype, type(alpha))
            if useapply_grad==0: #opt & applygrad:
                apply_grad(j, current, np.float32(alpha), grad_d,
                            wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)
                if useapply_grad: #opt & applygrad:
                    apply_grad(k, other, alpha, other_grad,
                                wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)
            elif useapply_grad==1:
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #current += alpha * grad_d
                for d in range(dim):
                    #current[d] += alpha * clip(grad_d[d])
                    current[d] += alpha * grad_d[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                if wrap_pt is not None:
                    wrap_pt(current)

                if move_other:
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(k, other, other_grad)
                    if wrap_grad is not None:
                        wrap_grad(other, other_grad)
                    #if _doclip:
                    #    other_grad = clip_array(other_grad)
                    #other += alpha * other_grad
                    for d in range(dim):
                        #other[d] += alpha * clip(other_grad[d])
                        other[d] += alpha * other_grad[d]
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(k, other)
                    if wrap_pt is not None:
                        wrap_pt(other)
            else:
                if move_other:
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                        wrap_idx_grad(k, other, other_grad)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                        wrap_grad(other, other_grad)
                    #if _doclip:
                    #    grad_d = clip_array(grad_d)
                    #    other_grad = clip_array(other_grad)
                    #current += alpha * grad_d
                    for d in range(dim):
                        #current[d] += alpha * clip(grad_d[d])
                        current[d] += alpha * grad_d[d]
                        other[d] += alpha * other_grad[d]
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                        wrap_idx_pt(k, other)
                    if wrap_pt is not None:
                        wrap_pt(current)
                        wrap_pt(other)
                else:
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                    #if _doclip:
                    #    grad_d = clip_array(grad_d)
                    #current += alpha * grad_d
                    for d in range(dim):
                        #current[d] += alpha * clip(grad_d[d])
                        current[d] += alpha * grad_d[d]
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                    if wrap_pt is not None:
                        wrap_pt(current)

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

                #if grad_coeff > 0.0:
                #    grad_d = clip_array(grad_coeff * (current - other))
                #else:
                #    # [ejk] seems strange
                #    #       Is this "anything to avoid accidental superpositions"?
                #    #       (why not randomly +/-4?)
                #    grad_d = np.full(dim, 4.0)
                for d in range(dim):
                    grad_d[d] = clip(grad_coeff * (current[d] - other[d])) if grad_coeff>0.0 else 4.0
                    other_grad[d] = -grad_d[d]
                    #if move_other:
                    #    other_grad[d] = - grad_d[d]

                #if move_other:
                #    other_grad = -grad_d.copy()
                #current_grad = grad_d
                if useapply_grad==0:
                    apply_grad(j, current, alpha, grad_d,
                                wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)
                    if move_other:
                        apply_grad(k, other, alpha, other_grad,
                                    wrap_idx_grad, wrap_grad, wrap_idx_pt, wrap_pt)
                elif useapply_grad==1:
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                    #if _doclip:
                    #    grad_d = clip_array(grad_d)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * grad_d[d]
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                    if wrap_pt is not None:
                        wrap_pt(current)

                    # following is needed for correctness if tail==head
                    #   upstream code does NOT do this
                    if move_other:
                        if wrap_idx_grad is not None:
                            wrap_idx_grad(k, other, other_grad)
                        if wrap_grad is not None:
                            wrap_grad(other, other_grad)
                        #if _doclip:
                        #   other_grad = clip_array(other_grad)
                        #other += alpha * other_grad
                        for d in range(dim):
                            other[d] += alpha * other_grad[d]
                        if wrap_idx_pt is not None:
                            wrap_idx_pt(k, other)
                        if wrap_pt is not None:
                            wrap_pt(other)

                else:
                    if move_other:
                        if wrap_idx_grad is not None:
                            wrap_idx_grad(j, current, grad_d)
                            wrap_idx_grad(k, other, other_grad)
                        if wrap_grad is not None:
                            wrap_grad(current, grad_d)
                            wrap_grad(other, other_grad)
                        #if _doclip: # but NEVER call clip_array (slow!)
                        #    grad_d = clip_array(grad_d)
                        #    other_grad = clip_array(other_grad)
                        #current += alpha * grad_d
                        for d in range(dim):
                            current[d] += alpha * grad_d[d]
                            other[d] += alpha * other_grad[d]
                        if wrap_idx_pt is not None:
                            wrap_idx_pt(j, current)
                            wrap_idx_pt(k, other)
                        if wrap_pt is not None:
                            wrap_pt(current)
                            wrap_pt(other)
                    else:
                        if wrap_idx_grad is not None:
                            wrap_idx_grad(j, current, grad_d)
                        if wrap_grad is not None:
                            wrap_grad(current, grad_d)
                        #if _doclip:
                        #    grad_d = clip_array(grad_d)
                        #current += alpha * grad_d
                        for d in range(dim):
                            current[d] += alpha * grad_d[d]
                        if wrap_idx_pt is not None:
                            wrap_idx_pt(j, current)
                        if wrap_pt is not None:
                            wrap_pt(current)

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
        # END if epoch_of_next_sample[i] <= n:
    # END for i in numba.prange(epochs_per_sample.shape[0]):

# NOT optimized for parallel=True
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
    wrap_idx_pt,    # NEW: constraint fn(idx,pt) or None
    wrap_idx_grad,  #      constraint fn(idx,pt,grad) or None
    wrap_pt,        #      constraint fn(pt) or None
    wrap_grad,      #      constraint fn(pt,grad) or None
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
    # I've taken the _apply_grad fully written out option here.
    # and will work on parallelism issues here.
    #
    #if constraints is None:
    #    constraints = {}

    # SUBJECT TO CHANGE ... should this be a user responsibility?
    #                       or handled in some other place?
    # WHEN CONSTRAINTS WERE FULLY-FLESHED-OUT PROJECTION OBJECTS ...
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
    #if wrap_idx_pt is not None: print("wrap_idx_pt",wrap_idx_pt)
    #if wrap_idx_grad is not None: print("wrap_idx_grad",wrap_idx_grad)
    #if wrap_pt is not None: print("wrap_pt",wrap_pt)
    #if wrap_grad is not None: print("wrap_grad",wrap_grad)

    # catch simplifying assumptions
    #   TODO: Work through constraint correctness when these fail!
    assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    assert np.all(head_embedding == tail_embedding)

    _doclip = wrap_idx_grad is not None or wrap_grad is not None

    # This ONLY works for numba parallel=False
    grad_d     = np.empty(dim, dtype=head_embedding.dtype) #one per thread!
    other_grad = np.empty(dim, dtype=head_embedding.dtype)
    for i in numba.prange(epochs_per_sample.shape[0]):
        #grad_cor_coeff = 0.0

        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            if densmap_flag:
                #grad_cor_coeff(dist_squared, a,b,
                #               dens_phi_sum, dens_re_sum, dens_re_cov, dens_re_std,
                #               dens_re_mean, dens_lambda, dens_R, dens_mu, dens_mu_tot)
                # Note that densmap could, if you wish, be viewed as a
                # "constraint" in that it's a "gradient modification function"
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

                # Reorganize double-clip of original umap impl to a simpler
                # densmap update of grad_coeff (result is a bit different, since
                # only a single 'clip' gets applied)
                grad_coeff += 2.0*grad_cor_coeff

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
            #delta = current - other # vector[dim]
            #grad_d = clip_array(grad_coeff * delta)
            #if densmap_flag:
            #    grad_d += clip_array(2 * grad_cor_coeff * delta)

            # NEW: can we elide zero-gradient?
            #   Yes: IF we have pre-applied to each point any point-positioning
            #        constraint functions AND we have no constraint-generated
            #        forces.
            # if Yes... could include a blocking
            #if ( dist_squared <= 0 # equiv grad_coeff==0.0
            #    and grad_cor_coeff == 0 ):
            # of code from here until just before the epoch_of_next_sample update

            #if False: # orig, grad_cor_coeff not rolled in to grad_coeff
            #    if densmap_flag:
            #        for d in range(dim):
            #            delta = current[d] - other[d]
            #            #grad_d[d] = clip(grad_coeff * delta)
            #            #grad_d[d] += clip(2.0 * grad_cor_coeff * delta)
            #            grad_d[d] = clip((grad_coeff + 2*grad_cor_coeff)
            #                             * (current[d]-other[d]))
            #            other_grad[d] = -grad_d[d]
            #    else:
            #        for d in range(dim):
            #            grad_d[d] = clip(grad_coeff * (current[d] - other[d]))
            #            other_grad[d] = -grad_d[d]
            #        #if move_other:
            #        #    other_grad[d] = -grad_d[d]
            #else:
            #    # do only a single clip and roll grad_cor_coeff into grad_cor during 'densmap' code
            #    # note: first calculating current-other might be a bit more
            #    #       tolerant of read-race conditions. (but maybe numba would fuse the loops?)
            #    for d in range(dim):
            #        grad_d[d] = clip(grad_coeff * (current[d] - other[d]))
            #        other_grad[d] = -grad_d[d]

            if move_other:
                for d in range(dim):
                    grad_d[d] = clip(grad_coeff * (current[d] - other[d]))
                    other_grad[d] = -grad_d[d]

                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                    wrap_idx_grad(k, other, other_grad)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                    wrap_grad(other, other_grad)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #    other_grad = clip_array(other_grad)
                #current += alpha * grad_d
                for d in range(dim):
                    #current[d] += alpha * clip(grad_d[d])
                    current[d] += alpha * grad_d[d]
                    other[d] += alpha * other_grad[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                    wrap_idx_pt(k, other)
                if wrap_pt is not None:
                    wrap_pt(current)
                    wrap_pt(other)
            else:
                for d in range(dim):
                    grad_d[d] = clip(grad_coeff * (current[d] - other[d]))

                # even if grad_coeff is 0.0, constraints MIGHT still cause
                # current to move, or
                # grad_d to become nonzero
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #current += alpha * grad_d
                for d in range(dim):
                    #current[d] += alpha * clip(grad_d[d])
                    current[d] += alpha * grad_d[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                if wrap_pt is not None:
                    wrap_pt(current)

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

                #if grad_coeff > 0.0:
                #    grad_d = clip_array(grad_coeff * (current - other))
                #else:
                #    # [ejk] seems strange
                #    #       Is this "anything to avoid accidental superpositions"?
                #    #       (why not randomly +/-4?)
                #    grad_d = np.full(dim, 4.0)
                #for d in range(dim):
                #    grad_d[d] = clip(grad_coeff * (current[d] - other[d])) if grad_coeff>0.0 else 4.0
                #    other_grad[d] = -grad_d[d]
                #    #if move_other:
                #    #    other_grad[d] = - grad_d[d]

                #if move_other:
                #    other_grad = -grad_d.copy()
                #current_grad = grad_d
                if move_other:
                    #
                    # Divergence:
                    #  move_other here is needed for correctness if tail==head,
                    #  but stock umap does NOT do any move_other for other random
                    #  samples.  Maybe to decrease bad things in parallel=True? Dunno.
                    #
                    #  If removed, the with constraints some clusters can go to
                    #  'outside' the constraint boundaries, when you really expect
                    #  unconstrainged points to lie between two 'extreme' embeddings
                    #  for a cluster.
                    #
                    for d in range(dim):
                        grad_d[d] = clip(grad_coeff * (current[d] - other[d])) if grad_coeff>0.0 else 4.0
                        other_grad[d] = -grad_d[d]
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                        wrap_idx_grad(k, other, other_grad)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                        wrap_grad(other, other_grad)
                    #if _doclip: # but NEVER call clip_array (slow!)
                    #    grad_d = clip_array(grad_d)
                    #    other_grad = clip_array(other_grad)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * grad_d[d]
                        other[d] += alpha * other_grad[d]
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                        wrap_idx_pt(k, other)
                    if wrap_pt is not None:
                        wrap_pt(current)
                        wrap_pt(other)
                else:
                    for d in range(dim):
                        grad_d[d] = clip(grad_coeff * (current[d] - other[d])) if grad_coeff>0.0 else 4.0
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                    #if _doclip:
                    #    grad_d = clip_array(grad_d)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * grad_d[d]
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                    if wrap_pt is not None:
                        wrap_pt(current)

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
        # END if epoch_of_next_sample[i] <= n:
    # END for i in numba.prange(epochs_per_sample.shape[0]):

# This one has tweaks for parallel=True
def _optimize_layout_euclidean_single_epoch_para_correct_but_slower(
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
    wrap_idx_pt,    # NEW: constraint fn(idx,pt) or None
    wrap_idx_grad,  #      constraint fn(idx,pt,grad) or None
    wrap_pt,        #      constraint fn(pt) or None
    wrap_grad,      #      constraint fn(pt,grad) or None
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
    # I've taken the _apply_grad fully written out option here.
    # and will work on parallelism issues here.
    #
    #if constraints is None:
    #    constraints = {}

    # SUBJECT TO CHANGE ... should this be a user responsibility?
    #                       or handled in some other place?
    # WHEN CONSTRAINTS WERE FULLY-FLESHED-OUT PROJECTION OBJECTS ...
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
    #if wrap_idx_pt is not None: print("wrap_idx_pt",wrap_idx_pt)
    #if wrap_idx_grad is not None: print("wrap_idx_grad",wrap_idx_grad)
    #if wrap_pt is not None: print("wrap_pt",wrap_pt)
    #if wrap_grad is not None: print("wrap_grad",wrap_grad)

    # catch simplifying assumptions
    #   TODO: Work through constraint correctness when these fail!
    assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    assert np.all(head_embedding == tail_embedding)

    _doclip = wrap_idx_grad is not None or wrap_grad is not None
    # Note: this COULD be used to clip grads AFTER grad-constraint functions

    # "usapply_grad==2" of the _dev version

    # numba threading notes:
    # ----------------------
    # env NUMBA_THREADING_LAYER or numba.config.THREADING_LAYER can be:
    #   'tbb', 'omp' or 'workqueue' (in that order of precedence)
    # We are NOT interested in spawn/fork/etc. from multiprocessing module.
    # numba.threading_layer()       prints threading layer, after a numbe parallel=True call
    # linux: probably you get 'omp', unless you 'conda install tbb'
    # numba.get/set_num_threads, or NUMBA_NUM_THREADS env or numba.config.NUMBA_NUM_THREADS
    #   set_num_threads: can only decrease (via mask) the number of threads.
    #                    set_num_threads before invoking the @njit(parallel=True) function
    # omp has get_thread_num, but numba doesn't abstract such a thing
    # python threads has threading.get_ident() or threading.current_thread().ident

    # The first question is how to move mallocs out of fast point-movement loop.
    # --------------------------------------------------------------------------
    # non-parallel would just do [outside the loop]:
    #grad_d     = np.empty(dim, dtype=head_embedding.dtype) #one per thread!
    #other_grad = np.empty(dim, dtype=head_embedding.dtype)
    # But then EVERY thread is overwriting everyone else's data all the time.
    # This would never work when parallelized.

    
    # The solution lies in using a 'detail' function that returns a thread number in {0,1,...}
    # For parallel=True, alloc enough workspace for all threads
    #thr = numba.get_num_threads() # if parallel, provide non-overlapped scratch areas
    #   Try avoid cache-thrash by sizing per-thread lines well apart
    #   let's provide an extra 4096 bytes, or 1024 floats to try to cache-isolate tmp vectors
    #tmpstride = 1024 if dim < 60 else 2*dim+1024
    #   each thread needs array mem for 2 dim-long vectors, padded to some longer length
    #tmp = np.empty( (thr, tmpstride), dtype=np.float32 ) 
    #tmp = np.empty( (thr, tmpstride), dtype=head_embedding.dtype ) 
    # Now tmp[tid*tmpstride, i] ~ grad_d[i] for i in [0,dim-1]
    # and tmp[tid*tmpstride, dim+i] ~ other_grad[i] for i in [0,dim-1]

    # The second question is how to properly support 'move_other':
    #
    #   One solution is to do assign work manually to different threads,
    #   but even this has issues in that numba does not expose any mutex
    #   interface from its threading backends (you could use cffi or
    #   ctypes, maybe, to call into C library code explicitly, maybe)
    #
    #   https://numba.discourse.group/t/how-to-use-locks-in-nopython-mode/221
    #   https://stackoverflow.com/questions/61372937/avoid-race-condition-in-numba
    #
    # A better variation on manual threading just has each thread accumulate
    # all the 'move_other' gradient info during the main loop,
    #   Then accumulate each thread's 'move_other' gradients.
    #   And finally (in parallel again) apply the accumulated move_other forces.
    # I.e. we use thread local tmp memory to rigorously avoid 'move_other'
    # clash.

    # Note that there is still some *read* clash still operating in hog-wild mode!
    #opt=0 # seems to work OK, but takes a slowdown wrt. explicit loops
    #opt=1 # np.empty work area within prange, explicit looping. DOES NOT WORK
    #opt=2 # also does not work.

    for i in numba.prange(epochs_per_sample.shape[0]):
        #
        # The 1st parallel=True avoids this malloc-inside-threaded-loop
        # Here we try more for correcting a HUGE parallelism flaw, and
        # getting something that has a chance to work (at least in a hog-wild
        # race sense)
        #
        #if opt==0: # easy solution: numpy vector ops
        #    pass
        #elif opt==1: # alloc inside loop, write out loops
        #if opt==1: #incorrect output
        # this is NOT one per thread, unfortunately
        grad_d     = np.empty(dim, dtype=head_embedding.dtype)
        other_grad = np.empty(dim, dtype=head_embedding.dtype)
        #else:
        #
        # for tbb and omp numba backends, this return a number in [0,1,2,...]
        #numba.np.ufunc._get_thread_id()
        #grad_d = tmp[ thr, 0:dim ]
        #other_grad = tmp[ thr, dim:(dim+dim) ]
        # ABOVE also broken!

        # Now every thread can use grad_d and other_grad without stomping on each other.
        #
        # Note: if item 'k' is the 'i' of some other thread (as in head equiv. to tail)
        # then even original umap code has a read-race as another thread *might* in mid-update
        # of 'k' (the 'i' of another thread might be our 'k')
        #
        # The move_other part, though, has even more likely races, since every item 'k'
        # might be the 'j' of some different thread.  Original umap elided the move_other
        # of 'k' always, for the second loop of sampled far-neighbors.  Maybe to reduce
        # race probability.
        #
        # On the other hand, "Hog Wild" updates, ignoring races are often quite successful
        # in practice.
        #
        # Fix:
        # j is in range head_embedding.shape[0]; k is in range tail_embedding.shape[0]
        # (i) accumulate into other_grads of shape == tail_embedding.shape
        # (ii) sum results of each threads other_grads
        # (iii) next loop applies move_other gradient

        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            if densmap_flag:
                #grad_cor_coeff(dist_squared, a,b,
                #               dens_phi_sum, dens_re_sum, dens_re_cov, dens_re_std,
                #               dens_re_mean, dens_lambda, dens_R, dens_mu, dens_mu_tot)
                # Note that densmap could, if you wish, be viewed as a
                # "constraint" in that it's a "gradient modification function"
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

                # Reorganize double-clip of original umap impl to a simpler
                # densmap update of grad_coeff (result is a bit different, since
                # only a single 'clip' gets applied)
                grad_coeff += 2.0*grad_cor_coeff

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
            #delta = current - other # vector[dim]
            #grad_d = clip_array(grad_coeff * delta)
            #if densmap_flag:
            #    grad_d += clip_array(2 * grad_cor_coeff * delta)
            #if True: #opt==0:
            # numpy vec-ops for temporaries (numba can handle this)
            #grad_d = current-other
            #for d in range(dim):
            #    grad_d[d] = clip(grad_coeff * grad_d[d])
            #other_grad = -grad_d
            #else:
            #    if densmap_flag:
            #        for d in range(dim):
            #            delta = current[d] - other[d]
            #            #grad_d[d] = clip(grad_coeff * delta)
            #            #grad_d[d] += clip(2.0 * grad_cor_coeff * delta)
            #            grad_d[d] = clip((grad_coeff + 2*grad_cor_coeff)
            #                             * (current[d]-other[d]))
            #            other_grad[d] = -grad_d[d]
            #    else:
            #        for d in range(dim):
            #            grad_d[d] = clip(grad_coeff * current[d] - other[d])
            #            other_grad[d] = -grad_d[d]
            #        #if move_other:
            #        #    other_grad[d] = -grad_d[d]
            #    # simplification (a little non-equivalent)
            #    # TODO: verify shape of grad_cor_coeff and do only a single clip ...


            #if move_other:
            #    other_grad = -grad_d.copy()
            #current_grad = grad_d
            #print(current.dtype, grad_d.dtype, type(alpha))
            if move_other:
                grad_d = current-other
                for d in range(dim):
                    grad_d[d] = clip(grad_coeff * grad_d[d])
                other_grad = -grad_d
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                    wrap_idx_grad(k, other, other_grad)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                    wrap_grad(other, other_grad)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #    other_grad = clip_array(other_grad)
                #current += alpha * grad_d
                for d in range(dim):
                    #current[d] += alpha * clip(grad_d[d])
                    current[d] += alpha * grad_d[d]
                    other[d] += alpha * other_grad[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                    wrap_idx_pt(k, other)
                if wrap_pt is not None:
                    wrap_pt(current)
                    wrap_pt(other)
            else:
                grad_d = current-other
                for d in range(dim):
                    grad_d[d] = clip(grad_coeff * grad_d[d])
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #current += alpha * grad_d
                for d in range(dim):
                    current[d] += alpha * clip(grad_d[d])
                    #current[d] += alpha * grad_d[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                if wrap_pt is not None:
                    wrap_pt(current)

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

                #if grad_coeff > 0.0:
                #    grad_d = clip_array(grad_coeff * (current - other))
                #else:
                #    # [ejk] seems strange
                #    #       Is this "anything to avoid accidental superpositions"?
                #    #       (why not randomly +/-4?)
                #    grad_d = np.full(dim, 4.0)
                #if True: #if opt==0:
                # numpy vec-ops for temporaries
                #grad_d = current-other
                #for d in range(dim):
                #    grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                #other_grad = -grad_d
                #else:
                #    for d in range(dim):
                #        grad_d[d] = clip(grad_coeff * (current[d] - other[d])) if grad_coeff>0.0 else 4.0
                #        other_grad[d] = -grad_d[d]
                #        #if move_other:
                #        #    other_grad[d] = - grad_d[d]
                

                #if move_other:
                #    other_grad = -grad_d.copy()
                #current_grad = grad_d
                if move_other:
                    # Note: stock umap NEVER does move_other for -ve samples
                    #
                    # Divergence:
                    #  move_other here is needed for correctness if tail==head,
                    #  but stock umap does NOT do any move_other for other random
                    #  samples.  Maybe to decrease bad things in parallel=True? Dunno.
                    #
                    grad_d = current-other
                    for d in range(dim):
                        grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                    other_grad = -grad_d
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                        wrap_idx_grad(k, other, other_grad)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                        wrap_grad(other, other_grad)
                    #if _doclip: # but NEVER call clip_array (slow!)
                    #    grad_d = clip_array(grad_d)
                    #    other_grad = clip_array(other_grad)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * clip(grad_d[d])
                        other[d] += alpha * clip(other_grad[d])
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                        wrap_idx_pt(k, other)
                    if wrap_pt is not None:
                        wrap_pt(current)
                        wrap_pt(other)
                else:
                    grad_d = current-other
                    for d in range(dim):
                        grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                    #if _doclip:
                    #    grad_d = clip_array(grad_d)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * clip(grad_d[d])
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                    if wrap_pt is not None:
                        wrap_pt(current)

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
        # END if epoch_of_next_sample[i] <= n:
    # END for i in numba.prange(epochs_per_sample.shape[0]):

# here np.empty inside prange, plus written-out 'dim' loops
def _optimize_layout_euclidean_single_epoch_para_correct_about_same_as_1cpu(
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
    wrap_idx_pt,    # NEW: constraint fn(idx,pt) or None
    wrap_idx_grad,  #      constraint fn(idx,pt,grad) or None
    wrap_pt,        #      constraint fn(pt) or None
    wrap_grad,      #      constraint fn(pt,grad) or None
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
    assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    assert np.all(head_embedding == tail_embedding)

    for i in numba.prange(epochs_per_sample.shape[0]):
        # Approach #1:
        # this is NOT one per thread, unfortunately
        grad_d     = np.empty(dim, dtype=head_embedding.dtype)
        other_grad = np.empty(dim, dtype=head_embedding.dtype)

        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            if densmap_flag:
                #grad_cor_coeff(dist_squared, a,b,
                #               dens_phi_sum, dens_re_sum, dens_re_cov, dens_re_std,
                #               dens_re_mean, dens_lambda, dens_R, dens_mu, dens_mu_tot)
                # Note that densmap could, if you wish, be viewed as a
                # "constraint" in that it's a "gradient modification function"
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

                # Reorganize double-clip of original umap impl to a simpler
                # densmap update of grad_coeff (result is a bit different, since
                # only a single 'clip' gets applied)
                grad_coeff += 2.0*grad_cor_coeff

            if move_other:
                #grad_d = current-other
                #for d in range(dim):
                #    grad_d[d] = clip(grad_coeff * grad_d[d])
                #other_grad = -grad_d
                for d in range(dim):
                    grad_d[d] = current[d] - other[d]
                    grad_d[d] = clip(grad_coeff * grad_d[d])
                    other_grad[d] = -grad_d[d]
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                    wrap_idx_grad(k, other, other_grad)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                    wrap_grad(other, other_grad)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #    other_grad = clip_array(other_grad)
                #current += alpha * grad_d
                for d in range(dim):
                    #current[d] += alpha * clip(grad_d[d])
                    current[d] += alpha * grad_d[d]
                    other[d] += alpha * other_grad[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                    wrap_idx_pt(k, other)
                if wrap_pt is not None:
                    wrap_pt(current)
                    wrap_pt(other)
            else:
                #grad_d = current-other
                #for d in range(dim):
                #    grad_d[d] = clip(grad_coeff * grad_d[d])
                for d in range(dim):
                    grad_d[d] = current[d] - other[d]
                    grad_d[d] = clip(grad_coeff * grad_d[d])
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                #if _doclip:
                #    grad_d = clip_array(grad_d)
                #current += alpha * grad_d
                for d in range(dim):
                    current[d] += alpha * clip(grad_d[d])
                    #current[d] += alpha * grad_d[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                if wrap_pt is not None:
                    wrap_pt(current)

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

                if move_other:
                    # Note: stock umap NEVER does move_other for -ve samples
                    #
                    # Divergence:
                    #  move_other here is needed for correctness if tail==head,
                    #  but stock umap does NOT do any move_other for other random
                    #  samples.  Maybe to decrease bad things in parallel=True? Dunno.
                    #
                    #grad_d = current-other
                    #for d in range(dim):
                    #    grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                    #other_grad = -grad_d
                    for d in range(dim):
                        grad_d[d] = current[d] - other[d]
                        grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                        other_grad[d] = -grad_d[d]
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                        wrap_idx_grad(k, other, other_grad)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                        wrap_grad(other, other_grad)
                    #if _doclip: # but NEVER call clip_array (slow!)
                    #    grad_d = clip_array(grad_d)
                    #    other_grad = clip_array(other_grad)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * clip(grad_d[d])
                        other[d] += alpha * clip(other_grad[d])
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                        wrap_idx_pt(k, other)
                    if wrap_pt is not None:
                        wrap_pt(current)
                        wrap_pt(other)
                else:
                    #grad_d = current-other
                    #for d in range(dim):
                    #    grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                    for d in range(dim):
                        grad_d[d] = current[d] - other[d]
                        grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                        other_grad[d] = -grad_d[d]
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                    #if _doclip:
                    #    grad_d = clip_array(grad_d)
                    #current += alpha * grad_d
                    for d in range(dim):
                        current[d] += alpha * clip(grad_d[d])
                    if wrap_idx_pt is not None:
                        wrap_idx_pt(j, current)
                    if wrap_pt is not None:
                        wrap_pt(current)

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
        # END if epoch_of_next_sample[i] <= n:
    # END for i in numba.prange(epochs_per_sample.shape[0]):

# use shared global workspace...
# this 'parallel' version finally IS faster for iris X[150,4] data
# It uses a work area shared between threads, and uses a numba internal
# to give 'current thread number' (ok for tbb or omp numba backends only)
def _optimize_layout_euclidean_single_epoch_para(
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
    wrap_idx_pt,    # NEW: constraint fn(idx,pt) or None
    wrap_idx_grad,  #      constraint fn(idx,pt,grad) or None
    wrap_pt,        #      constraint fn(pt) or None
    wrap_grad,      #      constraint fn(pt,grad) or None
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
    assert head_embedding.shape == tail_embedding.shape  # nice for constraints
    assert np.all(head_embedding == tail_embedding)

    nthr = numba.get_num_threads() # if parallel, provide non-overlapped scratch areas
    dimx = (2*dim+1024 + 1024-1) // 1024 # 2*dim pts rounded up to mult of 1024
    tmp = np.empty( (nthr, dimx), dtype=head_embedding.dtype ) # can do 2d or 1d tmp
    #tmp = np.empty( (nthr * dimx), dtype=head_embedding.dtype ) 

    # reclip gradients *after* gradient-modifying constraints?
    _grad_mods = wrap_idx_grad is not None or wrap_grad is not None

    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] > n:
            continue

        # Approach #1:
        # this is NOT one per thread, unfortunately
        #grad_d     = np.empty(dim, dtype=head_embedding.dtype)
        #other_grad = np.empty(dim, dtype=head_embedding.dtype)
        # This thread writes into these memory locations:
        thr = numba.np.ufunc._get_thread_id()   # for tbb or omp backed, gives {0,1,...}
        # try out "view" approach 1st:
        #grad_d = tmp[ thr, 0:dim ]
        #other_grad = tmp[ thr, dim:(dim+dim) ]
        #grad_d = tmp[ (thr*dimx):(thr*dimx+dim) ]
        #other_grad = tmp[ (thr*dimx+dim):(thr*dimx+2*dim) ]

        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]

        dist_squared = rdist(current, other)

        if dist_squared > 0.0:
            grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
            grad_coeff /= a * pow(dist_squared, b) + 1.0
        else:
            grad_coeff = 0.0

        if densmap_flag:
            #grad_cor_coeff(dist_squared, a,b,
            #               dens_phi_sum, dens_re_sum, dens_re_cov, dens_re_std,
            #               dens_re_mean, dens_lambda, dens_R, dens_mu, dens_mu_tot)
            # Note that densmap could, if you wish, be viewed as a
            # "constraint" in that it's a "gradient modification function"
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

            # Reorganize double-clip of original umap impl to a simpler
            # densmap update of grad_coeff (result is a bit different, since
            # only a single 'clip' gets applied)
            grad_coeff += 2.0*grad_cor_coeff

        if move_other:
            grad_d = tmp[ thr, 0:dim ]
            other_grad = tmp[ thr, dim:(dim+dim) ]
            #grad_d = current-other
            #for d in range(dim):
            #    grad_d[d] = clip(grad_coeff * grad_d[d])
            #other_grad = -grad_d
            if _grad_mods:
                for d in range(dim):
                    grad_d[d] = current[d] - other[d]
                    grad_d[d] = grad_coeff * grad_d[d] # without clip
                    other_grad[d] = -grad_d[d]
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                    wrap_idx_grad(k, other, other_grad)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                    wrap_grad(other, other_grad)
                for d in range(dim): # post-constraint gradient clip
                    grad_d[d] = alpha*clip(grad_d[d])
                    other_grad[d] = alpha*clip(other_grad[d])
            else:
                for d in range(dim):
                    grad_d[d] = current[d] - other[d]
                    grad_d[d] = alpha * clip(grad_coeff * grad_d[d])
                    other_grad[d] = -grad_d[d]
            #current += alpha * grad_d
            for d in range(dim):
                current[d] += grad_d[d]
                other[d] += other_grad[d]
            if wrap_idx_pt is not None:
                wrap_idx_pt(j, current)
                wrap_idx_pt(k, other)
            if wrap_pt is not None:
                wrap_pt(current)
                wrap_pt(other)
        else:
            grad_d = tmp[ thr, 0:dim ]
            if _grad_mods:
                for d in range(dim):
                    grad_d[d] = current[d] - other[d]
                    grad_d[d] = grad_coeff * grad_d[d] # elide clip here
                    other_grad[d] = -grad_d[d]
                if wrap_idx_grad is not None:
                    wrap_idx_grad(j, current, grad_d)
                    wrap_idx_grad(k, other, other_grad)
                if wrap_grad is not None:
                    wrap_grad(current, grad_d)
                    wrap_grad(other, other_grad)
                for d in range(dim): # post-constraint gradient clip
                    grad_d[d] = alpha*clip(grad_d[d])
                    other_grad[d] = alpha*clip(other_grad[d])
            else:
                for d in range(dim):
                    grad_d[d] = current[d] - other[d]
                    grad_d[d] = alpha * clip(grad_coeff * grad_d[d])
                    other_grad[d] = -grad_d[d]
            #current += alpha * grad_d
            for d in range(dim):
                current[d] += grad_d[d]
            if wrap_idx_pt is not None:
                wrap_idx_pt(j, current)
            if wrap_pt is not None:
                wrap_pt(current)

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

            if move_other:
                # Note: stock umap NEVER does move_other for -ve samples
                #
                # Divergence:
                #  move_other here is needed for correctness if tail==head,
                #  but stock umap does NOT do any move_other for other random
                #  samples.  Maybe to decrease bad things in parallel=True? Dunno.
                #
                #grad_d = current-other
                #for d in range(dim):
                #    grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                #other_grad = -grad_d
                if _grad_mods:
                    for d in range(dim):
                        grad_d[d] = current[d] - other[d]
                        grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                        other_grad[d] = -grad_d[d]
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                        wrap_idx_grad(k, other, other_grad)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                        wrap_grad(other, other_grad)
                    for d in range(dim): # also a post-constraint clip
                        current[d] += alpha * clip(grad_d[d])
                        other[d] += alpha * clip(other_grad[d])
                else:
                    for d in range(dim):
                        grad_d[d] = current[d] - other[d]
                        grad_d[d] = alpha * clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                        other_grad[d] = -grad_d[d]
                #current += alpha * grad_d
                for d in range(dim): # possible rare read-tear in parallel
                    current[d] += grad_d[d]
                    other[d] += other_grad[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                    wrap_idx_pt(k, other)
                if wrap_pt is not None:
                    wrap_pt(current)
                    wrap_pt(other)
            else:
                if _grad_mods:
                    for d in range(dim):
                        grad_d[d] = current[d] - other[d]
                        grad_d[d] = clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                        other_grad[d] = -grad_d[d]
                    if wrap_idx_grad is not None:
                        wrap_idx_grad(j, current, grad_d)
                    if wrap_grad is not None:
                        wrap_grad(current, grad_d)
                    for d in range(dim): # also a post-constraint clip
                        current[d] += alpha * clip(grad_d[d])
                else:
                    for d in range(dim):
                        grad_d[d] = current[d] - other[d]
                        grad_d[d] = alpha * clip(grad_coeff * grad_d[d]) if grad_coeff>0.0 else 4.0
                #current += alpha * grad_d
                for d in range(dim): # read-tear possible in parallel
                    current[d] += grad_d[d]
                if wrap_idx_pt is not None:
                    wrap_idx_pt(j, current)
                if wrap_pt is not None:
                    wrap_pt(current)

        epoch_of_next_negative_sample[i] += (
            n_neg_samples * epochs_per_negative_sample[i]
        )
        # END if epoch_of_next_sample[i] <= n:
    # END for i in numba.prange(epochs_per_sample.shape[0]):



opt_euclidean = numba.njit(_optimize_layout_euclidean_single_epoch,
                            fastmath=True, parallel=False,)

opt_euclidean_para = numba.njit(_optimize_layout_euclidean_single_epoch_para,
                                 fastmath=True, parallel=True, nogil=True)

@numba.njit()
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
    output_constrain=None, # independent of head_embedding index
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
        umap_.py simplicial_set_embedding always calls with move_other=True,
        even though our default is move_other=False
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    if False:
        print("optimize_layout_euclidean")
        print(type(head_embedding), head_embedding.dtype if isinstance(head_embedding, np.ndarray) else "")
        print(type(tail_embedding), tail_embedding.dtype if isinstance(tail_embedding, np.ndarray) else "")
        print(type(head), head.dtype if isinstance(head, np.ndarray) else "")
        print(type(tail), tail.dtype if isinstance(tail, np.ndarray) else "")
        print(type(n_epochs))
        print(type(epochs_per_sample), epochs_per_sample.dtype if isinstance(epochs_per_sample, np.ndarray) else "")
        print("a,b", type(a), type(b))
        print("rng_state", rng_state,type(rng_state))
        print("gamma, initial_alpha", type(gamma), type(initial_alpha))
        print("negative_sample_rate", type(negative_sample_rate))
        print("parallel, verbose, densmap", type(parallel) ,type(verbose), type(densmap))
        print("densmap_kwds", densmap_kwds)
        print("move_other", move_other)
        print("output_constrain", output_constrain)
        print("pin_mask", type(pin_mask))
        print("head,tail shapes", head_embedding.shape, tail_embedding.shape)

    dim = head_embedding.shape[1]
    #move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    # override for parallel debug:
    #if parallel==True:
    #    move_other = False
    #    print("parallel==True --> move_other=False")
    #print("euc parallel",parallel," move_other",move_other)

    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    if pin_mask is None:
        pin_mask = {}
    have_constraints = False
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
                have_constraints = True
            elif k=='idx_grad':
                fns_idx_grad = [pin_mask[k]]
                have_constraints = True
            #elif k=='epoch': fns_epoch = [pin_mask[k][0]]
            else:
                print(" Warning: unrecognized constraints key", k)
                print(" Allowed constraint keys:", recognized)
        # I'm trying to avoid numba 0.53 warnings (about firstclass functions)

    else:
        print("pin_mask", pin_mask.size, pin_mask.shape)
        assert (isinstance(pin_mask, np.ndarray))
        if len(pin_mask.shape) == 1:
            assert pin_mask.shape[0] == head_embedding.shape[0]
            # let's use pinindexed_grad for this one
            # (it's a more stringent test)
            idxs = []
            for i in range(pin_mask.size):
                if pin_mask[i]==0.0:
                    idxs.append(i)
            print("pin_mask 1d:",  len(idxs), "points don't move")
            fixed_pos_idxs = np.array(idxs)
            # todo [opt]: no-op if zero points were fixed
            @numba.njit()
            def pin_mask_1d(idx,pt,grad):
                return con.pinindexed_grad(idx,pt,grad, fixed_pos_idxs)
            fns_idx_grad = [pin_mask_1d,]
            have_constraints = True

        elif len(pin_mask.shape) == 2:
            assert pin_mask.shape == head_embedding.shape
            # DEBUG:
            #for i in range(pin_mask.shape[0]):
            #    for d in range(dim):
            #        if pin_mask[i,d] == 0.0:
            #            print("sample",i,"pin head[",d,"] begins at",head_embedding[i,d])
            # v.2 translates zeros in pin_mask to embedding values; nonzeros become np.inf
            # and then uses a 'freeinf' constraint from constraints.py
            # Current fixed point dimensions are copied into the constraint itself.
            freeinf_arg = np.where( pin_mask == 0.0, head_embedding, np.float32(np.inf) )
            print("pin_mask 2d:", np.sum(pin_mask==0.0), "dimensions of data set shape",
                  pin_mask.shape, "are held fixed")
            # original approach
            @numba.njit()
            def pin_mask_constraint(idx,pt):
                con.freeinf_pt( idx, pt, freeinf_arg )
            fns_idx_pt = [pin_mask_constraint,]
            have_constraints = True
            # OR mirror a tuple-based approach (REMOVED)
            #pin_mask_constraint_tuple = (con.freeinf_pt, freeinf_arg,)
            #@numba.njit()
            #def pin_mask_constraint2(idx,pt):
            #    pin_mask_constraint_tuple[0]( idx, pt, *pin_mask_constraint_tuple[1:] )
            #fns_idx_pt += (pin_mask_constraint2,)

        #assert pin_mask.shape == torch.Size(head_embedding.shape[0]))
        else:
            raise ValueError("pin_mask data_constrain must be a 1 or 2-dim array")

    # call a tuple of jit functions(idx,pt) in sequential order
    #   This did NOT work so well.  Possibly numba issues?
    #print("fns_idx_pt",fns_idx_pt)
    #print("fns_idx_grad",fns_idx_grad)
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

    outconstrain_pt = None
    outconstrain_grad = None
    outconstrain_epoch_pt = None
    outconstrain_final_pt = None
    if output_constrain is not None:
        assert isinstance(output_constrain, dict)
        for k in output_constrain:
            fn = output_constrain[k]
            if k == 'pt': outconstrain_pt = fn; have_constraints=True
            if k == 'grad': outconstrain_grad = fn; have_constraints=True
            if k == 'epoch_pt': outconstrain_epoch_pt = fn; have_constraints=True
            if k == 'final_pt': outconstrain_final_pt = fn; have_constraints=True

    if False and have_constraints:
        if wrap_idx_pt is not None: print("wrap_idx_pt",wrap_idx_pt)
        if wrap_idx_grad is not None: print("wrap_idx_grad",wrap_idx_grad)
        if outconstrain_pt is not None: print("outconstrain_pt",outconstrain_pt)
        if outconstrain_grad is not None: print("outconstrain_grad",outconstrain_grad)
        if outconstrain_epoch_pt is not None: print("outconstrain_epoch_pt",outconstrain_epoch_pt)
        if outconstrain_final_pt is not None: print("outconstrain_final_pt",outconstrain_final_pt)

    #print("euc parallel",parallel)
    #print("euc rng_state",rng_state,type(rng_state))

    #
    # TODO: fix numba errors with parallel=True
    #       There is an issue in numba with 
	# No implementation of function Function(<function runtime_broadcast_assert_shapes at 0x7f0bdd45be50>) found for signature:
	# 
	# >>> runtime_broadcast_assert_shapes(Literal[int](1), array(float64, 1d, C))
	# 
	#There are 2 candidate implementations:
	# - Of which 2 did not match due to:
	# Overload in function 'register_jitable.<locals>.wrap.<locals>.ov_wrap': File: numba/core/extending.py: Line 151.
	#   With argument(s): '(int64, array(float64, 1d, C))':
	#  Rejected as the implementation raised a specific error:
	#    TypeError: missing a required argument: 'arg1'
	#  raised from /home/ml/kruus/anaconda3/envs/miru/lib/python3.9/inspect.py:2977
    #
    #optimize_fn = numba.njit(
    #    _optimize_layout_euclidean_single_epoch, fastmath=True,
    #    #parallel=False,
    #    parallel=parallel,  # <--- Can this be re-enabled?
    #)
    #optimize_fn = _opt_euclidean
    optimize_fn = (opt_euclidean_para if parallel
                   else opt_euclidean)

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

    # Note: we do NOT adjust points to initially obey constraints.
    #       'init' conditions are up to the user, and might involve
    #       some best fit of an unconstrained UMAP to the constraints,
    #       followed by manually adjusting constrained points so
    #       they initially obey constraints.
    dens_re_std = 0
    dens_re_mean = 0
    dens_re_cov = 0
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
            outconstrain_pt,
            outconstrain_grad,
            #outconstrain_epoch_pt = None
            #outconstrain_final_pt = None
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

        # Should outconstrain_epoch_pt run before epoch loop too?
        if outconstrain_epoch_pt is not None:
            outconstrain_final_pt(head_embedding)
            # not sure move_other ...  XXX
            #if move_other:
            #    outconstrain_final_pt(tail_embedding)

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    if outconstrain_final_pt is not None:
        outconstrain_final_pt(head_embedding)
        #if move_other: # ... probably not
        #    outconstrain_final_pt(tail_embedding)

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

_opt_euclidean_aligned = numba.njit( _optimize_layout_aligned_euclidean_single_epoch,
                                    fastmath=True, parallel=False,)
_opt_euclidean_aligned_para = numba.njit(_optimize_layout_aligned_euclidean_single_epoch,
                                         fastmath=True, parallel=True,)

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

    #optimize_fn = numba.njit(
    #    _optimize_layout_aligned_euclidean_single_epoch,
    #    fastmath=True,
    #    parallel=parallel,
    #)
    optimize_fn = (_opt_euclidean_aligned_para if parallel
                   else _opt_euclidean_aligned)

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


