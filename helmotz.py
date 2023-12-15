import numpy as np
import matplotlib.pyplot as plt

class HelmholtzEquation:
    def __init__(self, domain_size, k):
        self.domain_size = domain_size
        self.k = k

    def exact_solution(self, x, y):
        # Simplified example of an exact solution for demonstration
        return np.sin(self.k * np.pi * x) * np.sin(self.k * np.pi * y)

    def sample_points(self, num_samples):
        # Generate sample points within the domain
        samples = np.random.uniform(-self.domain_size, self.domain_size, (num_samples, 2))
        values = self.exact_solution(samples[:, 0], samples[:, 1])
        return np.column_stack((samples, values))

    def plot_solution(self):
        # Create a grid of points
        x = np.linspace(-self.domain_size, self.domain_size, 200)
        y = np.linspace(-self.domain_size, self.domain_size, 200)
        X, Y = np.meshgrid(x, y)
        U = self.exact_solution(X, Y)

        # Plot the solution
        plt.figure(figsize=(8, 8))
        plt.contourf(X, Y, U, levels=15, cmap='bwr')
        plt.colorbar()
        plt.title('Helmholtz Equation Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Example usage
if __name__ == "__main__":
    domain_size = 1.0
    k = np.pi  # Wave number

    helmholtz = HelmholtzEquation(domain_size, k)
    helmholtz.plot_solution()

    # Sample points and their corresponding solution values
    samples = helmholtz.sample_points(1000)
    plt.scatter(samples[:, 0], samples[:, 1], c=samples[:, 2], cmap='bwr')
    plt.colorbar()
    plt.title('Sampled Points and Their Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #画个三维度
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=samples[:, 2], cmap='bwr')
    plt.show()
