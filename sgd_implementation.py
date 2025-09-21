import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, params, grads):
        for i, param in enumerate(params):
            if i not in self.velocities:
                self.velocities[i] = np.zeros_like(param)
            
            self.velocities[i] = self.momentum * self.velocities[i] + self.learning_rate * grads[i]
            params[i] -= self.velocities[i]
        return params

# Example usage:
if __name__ == '__main__':
    # Dummy parameters and gradients
    params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    grads = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    updated_params = optimizer.update(params, grads)

    print("Original Params:", params)
    print("Updated Params:", updated_params)