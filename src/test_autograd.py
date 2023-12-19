from neuralnetwork import MLP
from training import train
from graph import draw_dot
from datastructure import Value

def test_autograd():
    """
    Function to test the neural network.

    This function creates a neural network with 3 inputs, 4 neurons in each of the 2 hidden layers, and 1 output.
    It then trains the network using the given input data and desired targets.
    Ref: https://www.google.com/url?sa=i&url=https%3A%2F%2Fcs231n.github.io%2Fconvolutional-networks%2F&psig=AOvVaw0tMuYg0tapfXaaU7Aixq87&ust=1702997699209000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMj62PmemYMDFQAAAAAdAAAAABAD

    Args:
        None

    Returns:
        None
    """
    # xs = [[2.0, 3.0]]  # input data
    # ys = [1.0]  # desired targets
    # n = MLP(2, [3, 1]) # 2 inputs, 3 neurons in the hidden layer, 1 output   

    xs = [
        [2.0, 3.0, 1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

    n = MLP(3, [4, 4, 1])  # 3 inputs, 4 neurons in each of the 2 hidden layers, 1 output

    train(xs, ys, n)

def test_autograd_graph():
    # lets develop a neuron

    # inputs x1, x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of the neuron
    b = Value(6.8813735870195432, label='b')

    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'

    o = n.tanh(); o.label='o'
    o.backward()
    draw_dot(o)

def main():
    test_autograd()
    test_autograd_graph()

if __name__ == "__main__":
    main()
