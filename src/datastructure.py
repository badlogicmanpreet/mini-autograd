import math
import numpy as np
import random

class Value:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        """
        Initialize a Value object.

        Args:
            data: The data value associated with the node.
            _children: The children nodes of the current node.
            _op: The operation performed on the nodes.
            label: The label for the node.

        Returns:
            None
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        """
        Return a string representation of the Value object.

        Returns:
            A string representation of the Value object.
        """
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """
        Perform addition operation between two Value objects.

        Args:
            other: The other Value object to be added.

        Returns:
            The result of the addition operation as a new Value object.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        """
        Perform reverse addition operation between a Value object and another object.

        Args:
            other: The other object to be added.

        Returns:
            The result of the reverse addition operation as a new Value object.
        """
        return self + other
    
    def __mul__(self, other):
        """
        Perform multiplication operation between two Value objects.

        Args:
            other: The other Value object to be multiplied.

        Returns:
            The result of the multiplication operation as a new Value object.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward=_backward

        return out
    
    def __rmul__(self, other):
        """
        Perform reverse multiplication operation between a Value object and another object.

        Args:
            other: The other object to be multiplied.

        Returns:
            The result of the reverse multiplication operation as a new Value object.
        """
        return self * other
    
    def __neg__(self):
        """
        Perform negation operation on a Value object.

        Returns:
            The result of the negation operation as a new Value object.
        """
        return self * -1

    def __sub__(self, other):
        """
        Perform subtraction operation between two Value objects.

        Args:
            other: The other Value object to be subtracted.

        Returns:
            The result of the subtraction operation as a new Value object.
        """
        return self + (-other)

    def __pow__(self, other):
        """
        Perform exponentiation operation on a Value object.

        Args:
            other: The exponent value.

        Returns:
            The result of the exponentiation operation as a new Value object.
        """
        assert(isinstance(other, (int, float)))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        """
        Perform division operation between two Value objects.

        Args:
            other: The other Value object to be divided.

        Returns:
            The result of the division operation as a new Value object.
        """
        return self * other**-1

    def tanh(self):
        """
        Perform hyperbolic tangent operation on a Value object.

        Returns:
            The result of the hyperbolic tangent operation as a new Value object.
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward=_backward

        return out
    
    def exp(self):
        """
        Perform exponential operation on a Value object.

        Returns:
            The result of the exponential operation as a new Value object.
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out


    def backward(self):
        """
        Perform backpropagation to compute gradients for all nodes in the computation graph.

        Returns:
            None
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()