# Incidental Polysemanticity

This is the repository for the paper ["What Causes Polysemanticity? An Alternative Origin Story of Mixed Selectivity from Incidental Causes"](https://arxiv.org/abs/2312.03096). Check out the blog post [here](https://tmychow.com/posts/incidental_poly_0.html)!

Setup the environment by `pip install -r requirements.txt` and run the `evals.sh` script to reproduce the key results from the paper. It runs all of the scripts in `experiments`, which are:

- `all_curves.py` is for the aesthetic plot of polysemanticity curves for multiple modeld
- `sparsity.py` is for the plot on the speed of sparsification
- `fourth_norm.py` is for the plot on the sparsity from noise
- `collisions.py` is for the plot on the number of polysemantic neurons
- `correlations.py` is for the plot on the correlation between start and end weights