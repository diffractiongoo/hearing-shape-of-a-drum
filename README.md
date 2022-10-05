
# hearing-shape-of-a-drum
This repository gives the core codes for the paper: https://arxiv.org/abs/2203.08073.
## Constructing the environment

## Data generation
Four files are involved in the data generation process. 
- `generate_vertices.m` creates the Cartesian coordinates of the vertices as well as the innerangles of the polygons. 
- `compute_eigenvalues.nb` computes the first 100 Dirichlet eigenvalues for each polygon.
- `generate_grid.m` is called by `fill_matrix.m`. This file draws the boundaries of the polygons in a 41x41 grid.
- `fill_matrix.m` fills the boundaries with one. 
## Network construction and training
The following two files implement the networks depicted in the `Figure 3` of the main text.
- `train_image_predictor.py` constructs and trains the encoder-decoder network that predicts images of the pentagons from their eigenvalues. 
- `train_latent_analyzer.py` constructs and trains the latent analyzer network that mapes the latent space to physical parameters. 
