# Autograd Engine

Following along Andrej Karpathy's online course, I constructed an AutoGradient Engine that can dynamically construct directed acyclics graphs (DAGs) to perform binary classification using scalar-valued operations with a PyTorch-like API.

I then downloaded a file from an online public domain database (http://dramp.cpu-bioinfor.org/downloads/) associating peptide sequences with antimicrobial function, along with other related and auxiliary information. I parsed and pre-processed the amino acid sequences into one-hot encoders and constructed a deep neural network to train. I trained this AutoGrad-generated neural network on 70% of the data, saving 15% for validation and 15% for testing.

Credit: Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd"
