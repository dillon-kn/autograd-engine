# Same as micrograd.ipynb but in one file for export purposes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from graphviz import Digraph

RANDOM_STATE = 42

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data) # gradient
        self._backward = lambda: None # function for backward pass of gradient
        self._prev = set(_children) # set of parent nodes
        self._op = _op # operation used to calculate
        self.label = label # label for the node
    
    def __repr__(self):
        return f"Value(data={self.data})" # used to print object info
    
    def __add__(self, other):
        # Allow for adding of non-Value objects to Value objects
        other = other if isinstance(other, Value) else Value(other) # if other not Value, make it a Value
        out = Value(self.data + other.data, (self, other), '+') # create new Value object

        def _backward():
            # += because the nodes may be attached to multiple others
            self.grad += out.grad # just take on the gradient of the output, since it's addition
            other.grad += out.grad # same
        out._backward = _backward # assign backward function to other node's _backward attribute

        return out
    
    def __neg__(self):
        return self * -1 # return the additive inverse for subtraction purposes
    
    def __sub__(self, other):
        return self + (-other) # add the opposite

    def __mul__(self, other):
        # Allow for multiplication of non-Value objects to Value objects
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # += because the nodes may be attached to multiple others
            self.grad += other.data * out.grad # multiply gradient by the other data, which we treat as constant
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        # For when the left-side object is not a Value objectk
        return self * other
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Currently only supporting int or float" # make sure data type is correct for power
        out = Value(self.data ** other, (self,), f'**{other}') # create output Value object
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad # power rule + chain rule
        out._backward = _backward
        return out

    
    def exp(self): # This is e^x, not the exponential function
        x = self.data 
        out = Value(np.exp(x), (self,), 'exp') # Create new object for the output
        def _backward():
            self.grad += out.grad * out.data # e^x is in out, then apply chain rule
        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh()
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    
    def backward(self):
        topo = [] # init list to contain all nodes
        visited = set()
        # recursively build tree by adding node children to queue, adding current node to topo
        # then running build topo on each child in queue
        # note: it's a dfs construction
        def build_topo(v): 
            if v not in visited: 
                visited.add(v)
                for child in v._prev: 
                    build_topo(child) # run build topo on children
                topo.append(v) # add current node to topo
        build_topo(self)

        self.grad = 1.0
        # starting from the very first nodes, run backward function on each
        for node in reversed(topo):
            node._backward()


def trace(root):
    # build set of all nodes and edges in graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n)) # get mem address of node
        # for any value in graph, create rectangular record node for it
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad, ), shape='record')
        if n._op:
            # if value is result of operation, create op node for it
            dot.node(name=uid+n._op, label=n._op)
            # connect node to value
            dot.edge(uid+n._op, uid)
    for n1, n2 in edges:
        # connect n1 to op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    return dot

class Neuron:

    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)] # weights
        self.b = Value(np.random.uniform(-1, 1)) # bias

    def __call__(self, x):
        # dot product of weights and inputs + bias
        return sum((wi*xi for wi, xi in zip(self.w, x)), self.b).tanh()

class Layer:

    def __init__(self, nin, nout): # nin = # of inputs, # nout = # of neurons in layer
        self.neurons = [Neuron(nin) for _ in range(nout)] # create nout neurons, each with nin inputs

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons] # gets the activation of each neuron in the layer
        return outputs[0] if len(outputs) == 1 else outputs # if list has only one element, just return the element

class MLP: # multi-layer perceptron

    def __init__(self, nin, nouts): # nin = # of inputs, nouts = list of # of neurons in each layer
        size = [nin] + nouts # array of # of neurons in each layer
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nouts))] # create layers pairwise

    def __call__(self, x):
        for layer in self.layers: # feed forward through each layer
            x = layer(x) # get activations of each neuron in the layer
        return x
    
    def parameters(self):
        return [p for layer in self.layers for neuron in layer.neurons for p in neuron.w + [neuron.b]]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

def parse_data(filename="general_amps.xlsx"):
    # To validate amino acid sequencing data
    def is_valid_sequence(seq):
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        return all(char in valid_chars for char in seq.upper())

    # Load, parse, and preprocess data
    data = pd.read_excel(filename)

    # Filter out non-string sequences and invalid sequences
    valid_data = [(seq.upper(), label) for seq, label in zip(data['Sequence'], data['Activity']) 
                if isinstance(seq, str) and is_valid_sequence(seq)]

    # Split into sequences and labels
    sequences, labels = zip(*valid_data)
    sequences = [list(seq) for seq in sequences]
    labels = [1 if isinstance(label, str) and 'antimicrobial' in label.lower() else 0 for label in labels]

    # Find the maximum sequence length
    max_length = max(len(seq) for seq in sequences)

    # Identify all unique characters in the sequences
    unique_chars = sorted(set("ACDEFGHIKLMNPQRSTVWY"))
    unique_chars.append('')  # Add empty string for padding

    # Pad sequences to the same length for fixed input size to nn
    padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in sequences]

    # Encode sequences using one-hot encodings
    encoder = OneHotEncoder(sparse=False, categories=[unique_chars]*max_length)
    encoded_sequences = encoder.fit_transform(padded_sequences)

    # Split data into 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, test_size=0.3, random_state=RANDOM_STATE)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Split testing data into half validation half test, so 15% validation and 15% testing data
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

    return (X_train, y_train, X_val, X_test, y_val, y_test, encoded_sequences)

def create_model(encoded_sequences, hidden_size=64):
    # Define model size
    input_size = encoded_sequences.shape[1]
    output_size = 1  # anti-microbial or not
    model = MLP(input_size, [hidden_size, output_size]) # input_size -> 64 -> 1
    return model

# Define loss function
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 # for safe logarithmic calculations
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # keep y_pred between eps and 1-eps
    # Binary cross entropy loss function - common practice for binary classification
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Define training loop
def train(model, X, y, learning_rate=0.01, epochs=100, batch_size=32):
    X = np.array(X)
    y = np.array(y)
    
    for epoch in range(epochs):
        # Create mini-batches
        permutation = np.random.permutation(len(X))
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            y_pred = np.array([model(x)._data for x in X_batch])

            # Compute loss
            loss = binary_cross_entropy(y_batch, y_pred)

            # Backward pass
            model.zero_grad()
            for j in range(len(X_batch)):
                model(X_batch[j]).backward()

            # Update weights
            for p in model.parameters():
                p.data -= learning_rate * p.grad / len(X_batch)


        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

# Function to evaluate accuracy
def evaluate(model, X, y):
    correct = 0
    total = len(X)
    for x, y_true in zip(X, y):
        y_pred = model(x)
        if (y_pred.data > 0.5) == y_true:
            correct += 1
    return correct / total

def main():
    X_train, y_train, X_val, X_test, y_val, y_test, encoded_sequences = parse_data()

    model = create_model(encoded_sequences)

    # Train model
    train(model, X_train, y_train)

    # Evaluate model
    train_accuracy = evaluate(model, X_train, y_train)
    val_accuracy = evaluate(model, X_val, y_val)
    test_accuracy = evaluate(model, X_test, y_test)

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")