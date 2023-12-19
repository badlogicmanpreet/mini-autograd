from datastructure import Value
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Any

class Neuron:
    """
    A class representing a single neuron in a neural network.

    Attributes:
        w (list): List of weights for each input.
        b (Value): Bias value.

    Methods:
        __init__(nin): Initializes the neuron with random weights and bias.
        __call__(x): Performs the forward pass of the neuron.
        parameters(): Returns the list of parameters (weights and bias) of the neuron.
    """

    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    """
    Represents a layer in a neural network.

    Args:
        nin (int): Number of inputs to a single neuron.
        nout (int): Number of neurons in the layer.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Passes the input 'x' through each neuron in the layer and returns the outputs.

        Args:
            x: Input to the layer.

        Returns:
            The output of each neuron in the layer.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """
        Returns a list of parameters for the layer.

        Returns:
            A list of parameters for each neuron in the layer.
        """
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

class MLP:
    """
    Multi-Layer Perceptron (MLP) class.
    
    Args:
        nin (int): Number of input neurons.
        nouts (list): List of integers representing the number of neurons in each layer.
    
    Attributes:
        layers (list): List of Layer objects representing the layers in the MLP.
    """
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
