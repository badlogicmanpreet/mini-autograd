def train(xs, ys, n):
    """
    Performs training using forward and backward passes to optimize the neural network parameters.

    Returns:
        None
    """
    for k in range(100):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

        # backward pass
        for p in n.parameters(): # ensuring the grad is flushed before every backward pass (a.k.a zero grad)
            p.grad = 0.0
        loss.backward()

        # change
        for p in n.parameters():
            p.data += -0.1 * p.grad

        print(k, loss)

    print("Predictions:", ypred)

