import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]
        return params

# Example usage:
if __name__ == '__main__':
    # Dummy parameters and gradients
    params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    grads = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    optimizer = SGD(learning_rate=0.01)
    updated_params = optimizer.update(params, grads)

    print("Original Params:", params)
    print("Updated Params:", updated_params)