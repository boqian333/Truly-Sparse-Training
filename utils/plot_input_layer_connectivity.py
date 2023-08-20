import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# update the data for each frame
def anim(n):
    global data
    global allConnections
    data = allConnections[:,:,n]
    imobj.set_array(data)
    return imobj,



for i in range(1):
    data = np.load("../data/fashion_mnist.npz")
    connections=np.load("../Pretrained_results/set_mlp_5000_training_samples_e13_rand"+str(i)+"_input_connections.npz")["inputLayerConnections"]

    allConnections=np.zeros((28,28,len(connections)))
    for j in range(len(connections)):
        connectionsEpoch=np.reshape(connections[j],(28,28))
        allConnections[:,:,j]=connectionsEpoch

    fig = plt.figure()
    fig.suptitle('6th International (online) Summer school on AI and Big Data \nScalable Deep Learning Tutorial', fontsize=14)

    ax1 = fig.add_subplot(121)
    ax1.imshow(np.reshape(data["X_train"][1,:],(28,28)),vmin=0,vmax=255,cmap="gray_r",interpolation=None)
    ax1.set_title("Fashion-MNIST example")

    ax2 = fig.add_subplot(122)
    data=allConnections[:,:,0]
    imobj = ax2.imshow(data,vmin=0,vmax=np.max(allConnections),cmap="jet",interpolation=None)
    ax2.set_title("Input connectivity pattern evolution\nwith SET-MLP")

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar=fig.colorbar(imobj,cax=cax)
    cbar.set_label('Connections per input neuron (pixel)',size=8)

    fig.tight_layout()

    # create the animation
    ani = matplotlib.animation.FuncAnimation(fig, anim, frames=len(connections))
    ani.save("../Results/fashion_mnist_connections_evolution_per_input_pixel_rand"+str(i)+".gif", writer='imagemagick',fps=24,codec=None)