# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt

class Msdnn2():
    def __init__(self, subs_number):
        self.subs_number = subs_number
        self.layer_sizes = [2, 3, 3, 3, 1]
        self.node_positions = {}
        self.final_2 = None  # 倒数第二层的4个节点
        self.calculate_positions()
        self.final=[subs_number,1] #4 is the number of scales and 1 is the number of nodes in the final layer

    def calculate_positions(self):
        """
        Calculates the positions of each neuron in each layer and scale adaptively.
        """

        # Determine the vertical space needed by the largest layer
        max_layer_size = max(self.layer_sizes)
        layer_height = 20 / max_layer_size  # Adaptive height based on the largest layer

        # Determine the spacing between scales adaptively
        scale_spacing = 80.0 / self.subs_number

        # Calculate positions for each layer and scale
        for sub in range(self.subs_number):
            for layer_idx, size in enumerate(self.layer_sizes):
                positions = self.layer_positions(size, layer_idx, layer_height, scale_spacing * sub)
                self.node_positions[f'scale{sub + 1}_layer{layer_idx}'] = positions
        # Calculate the positions of the second-to-last layer and final node adaptively
        self.final_2 = np.array([[self.subs_number, scale_spacing * sub] for sub in range(self.subs_number)])
        positions = np.array([[len(self.layer_sizes) + 1.0, scale_spacing * (self.subs_number -2.5)]])
        print(positions)
        self.node_positions['final_out'] = positions

    def layer_positions(self, n_nodes, layer_idx, max_height,scale_index):
        """
        Calculate the positions for a layer of nodes.
        """
        if n_nodes == 1:
            return np.array([[layer_idx, scale_index]])
        else:
            return np.column_stack((np.full(n_nodes, layer_idx), scale_index+np.linspace(-max_height , max_height, n_nodes)))

    def draw_connections_with_activation(self, layer1_positions, layer2_positions, ax):
        """
        Draws connections between two layers.
        """

        for i, pos1 in enumerate(layer1_positions):
            for j, pos2 in enumerate(layer2_positions):
                weight = 0  # Since weights are initialized to 0
                arrow_props = dict(arrowstyle="-",color='blue', alpha=0.75, lw=2)

                ax.annotate("", xy=pos2, xytext=pos1, arrowprops=arrow_props)

    def draw_nodes(self, ax):
        """
        Draws the nodes on the given axes.
        """
        for key, positions in self.node_positions.items():

            for pos in positions:

                circle = plt.Circle(pos, 0.01, color='black', zorder=1)
                ax.add_artist(circle)

            ax.scatter(positions[:, 0],
                       positions[:, 1],
                       s=50,
                       color='black'
                      , zorder=4)

    def draw(self):
        """
        Draws the entire neural network.
        """


        fig_width = 6 + self.subs_number  # Adjust the width based on the number of subnetworks
        fig_height = 6  # Keep the height constant or adjust as necessary

        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        # Draw connections and nodes for each scale and layer

        for sub in range(self.subs_number):
            for layer_idx in range(len(self.layer_sizes) - 1):
                layer1_key = f'scale{sub + 1}_layer{layer_idx}'
                layer2_key = f'scale{sub + 1}_layer{layer_idx + 1}'
                if layer1_key in self.node_positions and layer2_key in self.node_positions:
                    self.draw_connections_with_activation(
                        self.node_positions[layer1_key],
                        self.node_positions[layer2_key],
                        ax
                    )

        # Draw connections between scales
        self.draw_connections_with_activation(
            self.final_2,
            self.node_positions["final_out"],
            ax
        )
        # Save the figure with high DPI
        plt.savefig('neural_network.png', dpi=300)  # Use a high DPI for better clarity
        # Adjust the display limits dynamically
        all_positions = np.concatenate(list(self.node_positions.values()))
        x_min, y_min = np.min(all_positions, axis=0)
        x_max, y_max = np.max(all_positions, axis=0)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 20, y_max + 20)  # Adding some padding for aesthetics
        self.draw_nodes(ax)
        ax.axis('off')
        plt.show()



# Creating an instance of the Msdnn2 class and drawing the neural network
ms = Msdnn2(4)
ms.draw()

