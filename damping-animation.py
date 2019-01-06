import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation


np.random.seed(8)

size = 300
node_size = 10000/size
arrow_size = min(1500/size, 15)
sparsity = 1.5/np.sqrt(size)

#damping = 0.15
#damping_list = list(np.linspace(0.01, 0.2, 20)) + list(np.linspace(0.19, 0.02, 18))
#damping_list = [1/10**x for x in np.arange(0.5, 2.25, 0.125)]
upper = 2
lower = 0
ratio = (upper-lower)/2
add = (upper+lower)/2
xs = [(ratio * np.cos(y)) + add for y in np.linspace(0, 3.14, 20)]
damping_list = [1/10**x for x in xs]

damping_list = list(np.linspace(0,1,20))

damping_list += damping_list[1:-1][::-1]

# GRAPH 1
# size = 50
# sparsity = 0.15

# alpha = 41
# beta = 54
# gamma = 5
# alpha, beta, gamma = (val / (alpha + beta + gamma) for val in [alpha, beta, gamma])
# G = nx.scale_free_graph(size, alpha=alpha, beta=beta, gamma=gamma, 
#                                            delta_in=0.2, delta_out=0)
# pos = nx.layout.spring_layout(G, iterations=10)
# A = A = nx.to_numpy_matrix(G, weight=1)

# GRAPH 2
G = nx.random_geometric_graph(size, sparsity, seed=8).to_directed()
#G = nx.random_geometric_graph(5, 0.7).to_directed()
new = []
mix = [(x,y) for x,y in np.random.permutation(G.edges)]
for edge in mix:
#for edge in G.edges:
    x, y = edge[0], edge[1]
    cond = (y,x) in new
    if not cond:
        new.append((x,y))
G.remove_edges_from(new)
pos = nx.get_node_attributes(G, 'pos')



A = nx.to_numpy_matrix(G)

D = np.divide(A,A.sum(axis=1),out=np.zeros_like(A), where=A!=0)
# For dangling nodes add restart probabilities
indicator = np.ones((size,1)) - D.sum(axis=1)
S = (1/size) * (indicator @ np.ones((1,size)))
D = D+S
B = np.ones(shape=(size,size)) / size

fig, ax = plt.subplots(1,1,figsize=(8,8))

def update(num):
    ax.clear()
    damping = damping_list[num]
    #print(num)
    print(round(damping, 5))

    M = (1-damping)*D + damping*B
    
    #print('Google Network:')
    #print(M.round(2))
    #print('\n')
    
    v_last = np.ones((1,size)) / np.ones((1,size)).sum()
    v_curr = np.ones((1,size)) / np.ones((1,size)).sum()
    diffs = []
    diff = 1
    epsilon = 0.001
    while diff > epsilon:
        v_curr = (v_last @ M)
        diff = np.sum(np.abs((v_curr - v_last)))
        diffs += [diff]
        v_last = v_curr
    v_last = np.array(v_last)[0]
    
    # Plot Network
    #G = nx.DiGraph(A)
    
    #pos = nx.layout.kamada_kawai_layout(G)
    #pos = nx.layout.spring_layout(G, iterations=10)
    #nx.draw_networkx_nodes(G, pos, node_color='Red', ax=axes[0], node_size=node_size)
    nx.draw_networkx_nodes(G, pos, node_color='Red', ax=ax, node_size=((node_size/v_last.mean())*v_last),
                           #node_color=v_last,
                          #node_cmap=plt.cm.Reds
                          )
    #nx.draw_networkx_edges(G, pos, #edge_color='Gray', 
    #                       alpha=0.5,
    #                       node_size=node_size,
    #                       arrowsize=arrow_size, width=1, ax=axes[0])
    nx.draw_networkx_edges(G, pos, 
                           #edge_color='Gray', 
                           alpha=0.5,
                           node_size=((node_size/v_last.mean())*v_last),
                           arrowsize=arrow_size, width=1, ax=ax)
    #axes[0].set_axis_off()
    ax.set_axis_off()
    ax.set_title(f'damping={round(damping,3)}')
    #fig.savefig(f'test_network{i}.pdf')
    #plt.show()
    
    
anim = FuncAnimation(
    fig, update, 
    interval=50, 
    frames=len(damping_list)
)

anim.save('damping.mp4')
 
plt.draw()
plt.show()