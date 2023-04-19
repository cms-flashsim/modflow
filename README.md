# modflow
This is our modding of the nflows package to implement a stricter identity init and avoid numerical errors.

### Install 

To use it clone the repo, `cd ./modflow` and the

``` py
pip install -e modflow
```
or if using from a colab you should be able to pip install the main branch as

```
pip install "modflow @git+https://github.com/cms-flashsim/modflow"
```

### Use

At the moment we have modded three classes of transform. They are the AutoregressiveAffine, AutoregressiveRQSpline, CouplingRQSpline. To use them install the package and then do:

```py 
from modflow.transformers import (MaskedAffineAutoregressiveTransform,
MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
PiecewiseRationalQuadraticCouplingTransform)
```

Compared to nflows they have only one additional input argument `init_identity` which is `True` by default.