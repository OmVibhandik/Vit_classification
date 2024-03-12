H = 256
W = 256
C = 3
P = 16

# Initially it was H*W*C
# Converted to N*(P^2)*C

N = H*W/P**2  # Input Seq Length
# Then converted to N*D with a Linear Layer

# D is Latent Vector Size

# Then add a class Embedding at the start
# Then positional Embeddings are appended

# This goes to LayerNorm
# Then to MSA
# Then residual connection
# Then again LayerNorm
# Then MLP Block with residual from 3rd point
# The output of class embedding will give the class of the image
# The positonal embeddings has to be learned from scratch

# For fine-tuning zero-initializes DxK FF layer
