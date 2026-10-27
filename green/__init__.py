"""Green: a proof-of-compute blockchain."""

import os

# A work unit is a few small matrix products. A BLAS thread per core makes each one
# slower and serialises the threads that check and mine units (measured on 20 cores:
# one BLAS thread, 2.3 ms per unit alone and 1.35 ms each across 8 threads; a thread
# per core, 2.6 ms alone and no faster across 8). The BLAS reads this once, when
# numpy loads, so it is set here, before; an operator's own setting wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
del _var
