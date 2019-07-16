import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# File to read
file_mem_usage = "./mem_usage.dat"

# Data to show
number_data = 1800

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    mem_data = np.loadtxt(file_mem_usage, dtype=np.float)

    # Add x and y to lists
    xs = np.arange(0, 2*mem_data.size, 2)
    ys = mem_data

    # Limit x and y lists to 20 items
    xs = xs[-number_data:]
    ys = ys[-number_data:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xlabel('time in sec')
    plt.ylabel('%RAM')
    plt.title('RAM usage. Total RAM = 504GB')

if len(sys.argv) != 2:
	print("This script needs one and only one additional argument, either 'final' or 'animated'.")
elif sys.argv[1] == 'animated':
	ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=5000)
	plt.show()
elif sys.argv[1] == 'final':
	mem_data = np.loadtxt(file_mem_usage, dtype=np.float)

	xs = np.arange(0, 2*mem_data.size, 2)
	ys = mem_data

	ax.plot(xs, ys)
	plt.xlabel('time in sec')
	plt.ylabel('%RAM')
	plt.title('RAM usage. Total RAM = 504GB')

	plt.savefig('./mem_usage.pdf')


