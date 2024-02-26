import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .problems import demo_func

def convergence_plot(x_hist, y_hist, filename, plot_title = 'Convergence Plot with gbest Position'):
    # Create convergence plot
    plt.figure(figsize=(8, 6))
    plt.plot(y_hist, marker='o', linestyle='-')
    plt.title(plot_title)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)

    # Add coords gbest as label for each points
    prev_coords = None
    for i, coords in enumerate(x_hist):
        if prev_coords is None or not np.all(coords == prev_coords):  # If coord not eq prev, add label
            formatted_coords = "( {:.2f}, {:.2f} )".format(coords[0], coords[1])
            plt.text(i, y_hist[i], f'({formatted_coords})', fontsize=8, ha='center', va='top')
            prev_coords = coords
    
    # Save convergence plot as image
    plt.savefig(filename)
    plt.close()

def particle_animation(record_value, filename, frames):
    # Create animation particles
    X_list, V_list = record_value['X'], record_value['V']

    fig, ax = plt.subplots(1, 1)
    ax.set_title('title', loc='center')
    line = ax.plot([], [], 'b.')

    X_grid, Y_grid = np.meshgrid(np.linspace(-3.0, 3.0, 40), np.linspace(-3.0, 3.0, 40))
    Z_grid = demo_func((X_grid, Y_grid))
    ax.contour(X_grid, Y_grid, Z_grid, 30)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    def update_scatter(frame):
        i, j = frame // 10, frame % 10
        ax.set_title('iter = ' + str(i))
        X_tmp = X_list[i] + V_list[i] * j / 10.0
        plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
        return line
    
    ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=frames * 10)

    ani.save(filename, writer='pillow', fps=24)
    plt.close()

def problem_plot_3d(filename):
    X_grid, Y_grid = np.meshgrid(np.linspace(-3.0, 3.0, 40), np.linspace(-3.0, 3.0, 40))
    Z_grid = demo_func((X_grid, Y_grid))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='seismic', edgecolor='none' )
    ax.contour(X_grid, Y_grid, Z_grid, 30, colors= 'black')
    
    # Save 3d plot as image
    plt.savefig(filename)
    plt.close()