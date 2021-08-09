import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_node(3)

G.add_edge(1,2)
G.add_edge(1,3)

def new_connection(graph):
    total_nodes = len(graph.nodes)
    graph.add_node(total_nodes+1)
    graph.add_edge(1, total_nodes+1, color="green")
    return

print(G.nodes)
new_connection(G)
new_connection(G)
new_connection(G)

print(G.nodes)

colors = []
for node in G.nodes:
    colors.append("b")
colors[0] = "g"
print(colors)
nx.draw_circular(G, node_color=colors)



plt.show()
