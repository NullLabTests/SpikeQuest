import matplotlib.pyplot as plt

def simulate_process(steps):
    """
    Simulates a process where a value increases in steps at specific points.
    
    Args:
        steps (int): Number of time steps to simulate.
    
    Returns:
        tuple: Lists of x (steps) and y (cumulative value) coordinates.
    """
    x = [i * 0.2 for i in range(steps)]  # Scale x to match 0.0 to 4.0 range over 20 steps
    y = []
    current_value = 0.0
    step_points = {2: 0.5, 7: 1.0, 12: 2.0, 15: 4.0}  # Step indices and values
    
    for i in range(steps):
        if i in step_points:
            current_value = step_points[i]
        y.append(current_value)
        
        # Simulate computational delay (optional, mimics real processing time)
        plt.pause(0.1)
        
    return x, y

def plot_results(x, y):
    """
    Plots the simulation results as a step-like line graph.
    
    Args:
        x (list): X-axis values (steps).
        y (list): Y-axis values (cumulative value).
    """
    plt.plot(x, y, 'b-', label='Progress')
    plt.xlabel('Steps (scaled)')
    plt.ylabel('Cumulative Value')
    plt.title('Figure 1: Process Progress Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    total_steps = 20  # One more than 19 to include step 19 (0-based indexing)
    
    try:
        # Run the simulation
        x_values, y_values = simulate_process(total_steps)
        
        # Check if the goal is reached (final value matches the graph's max)
        if y_values[-1] == 4.0:
            print("Goal reached at step 19!")
        
        # Plot the results
        plot_results(x_values, y_values)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        # Plot whatever was completed before interruption
        if 'x_values' in locals() and 'y_values' in locals():
            plot_results(x_values[:len(y_values)], y_values)
