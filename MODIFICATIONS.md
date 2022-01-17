## This is based on [umap PR 620](https://github.com/lmcinnes/umap/pull/620)
- Original PR touched 3 files:
  - `layouts.py` ~ line 184
    - add routine `_optimize_layout_euclidean_masked_single_epoch`
    - add routine `optimize_layout_euclidean_masked`
  - `parametric_umap.py` ~ line 374
    - add `pin_mask` arg to `_fit_embed_data`
      - it must be None: warn that it's unsupported
  - `umap_.py`: many changes to support `pin_mask`

- This fork wants to support 2-D gradient mask.

- Consider also borrowing PyMDE's **_constraint_** viewpoint,  
as a projection of points (or gradients)  
onto some _manifold_ (or tangent plane)

## todo
- I suspect I'd rather use a dense hi-D representation, because
  its distance metric might be more "diffusion-ready".   If sparse,
  too many things might overlap in "only a few ways" so "diffusion"
  is less well defined. (connectivity can be too coarse-grained).

- ? how to handle time-dependence?
  i.e. interval events (text passages, movie segments)
