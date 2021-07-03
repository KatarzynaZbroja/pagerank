import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import array
import subprocess
import shlex
import pandas as pd


def create_files():
    f = open('zbior.txt', 'w')
    f.write('#FromNodeId' + '\t'+ 'ToNodeId' + '\t' + 'Weight' + '\n')

    for axx in range(1,100001):
        axx = random.randint(1,100001)
        axy = random.randint(1,100001)
        axz = random.randint(1,100)
        axz = axz / 100
        f.write(str(axx) + '\t'+ str(axy) + '\t' + str(axz) + '\n')
    f.close()

    f = open('waga.txt', 'w')
    f.write('#FromNodeId' + '\t'+ 'ToNodeId' + '\t' + 'Weight' + '\n')

    for axx in range(1,21):
        axx = random.randint(1,21)
        axy = random.randint(1,21)
        axz = random.randint(1,100)
        axz = axz / 100
        f.write(str(axx) + '\t'+ str(axy) + '\t' + str(axz) + '\n')
    f.close()
    return 0

def pagerank():			
	G = nx.read_edgelist('./zbior.txt',nodetype=int,
		data=(('weight',float),), create_using=nx.DiGraph())
	alpha=0.85
	
	personalization=None
	
	max_iter=100
	
	tol=1.0e-6
	
	nstart=None
	
	weight='weight'
	
	dangling=None
	
	x_ax = []
	y_ax = []
			
		
	if len(G) == 0:
		return {}

	if not G.is_directed():
		D = G.to_directed()
	else:
		D = G
	W = nx.stochastic_graph(D, weight=weight)
	N = W.number_of_nodes()
	if nstart is None:
		axx = dict.fromkeys(W, 1.0 / N)
	else:
		s = float(sum(nstart.values()))
		axx = dict((k, v / s) for k, v in nstart.items())

	if personalization is None:
		p = dict.fromkeys(W, 1.0 / N)
	else:
		s = float(sum(personalization.values()))
		p = dict((k, v / s) for k, v in personalization.items())

	if dangling is None:
		dangling_weights = p
	else:
		s = float(sum(dangling.values()))
		dangling_weights = dict((k, v/s) for k, v in dangling.items())
	dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
	
	for zdd in range(max_iter):
		x_ax.append(zdd)
		xlast = axx
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in axx:
			for nbr in W[n]:
				axx[nbr] += alpha * xlast[n] * W[n][nbr][weight]
			axx[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
		err = sum([abs(x[n] - xlast[n]) for n in axx])
		y_ax.append(err)
		if err < N*tol:
			return print(x)
           
def pagerank_chart():			
	G = nx.read_edgelist('./zbior.txt',nodetype=int,
		data=(('weight',float),), create_using=nx.DiGraph())
	alpha=0.85
	personalization=None
	max_iter=100
	tol=1.0e-6
	nstart=None
	weight='weight'
	dangling=None
	
	xdd = []
	ydd = []
				
	if len(G) == 0:
		return {}

	if not G.is_directed():
		D = G.to_directed()
	else:
		D = G
	W = nx.stochastic_graph(D, weight=weight)
	N = W.number_of_nodes()
	if nstart is None:
		x = dict.fromkeys(W, 1.0 / N)
	else:
		s = float(sum(nstart.values()))
		x = dict((k, v / s) for k, v in nstart.items())

	if personalization is None:
		p = dict.fromkeys(W, 1.0 / N)
	else:
		s = float(sum(personalization.values()))
		p = dict((k, v / s) for k, v in personalization.items())

	if dangling is None:
		dangling_weights = p
	else:
		s = float(sum(dangling.values()))
		dangling_weights = dict((k, v/s) for k, v in dangling.items())
	dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
	for zdd in range(max_iter):
		xdd.append(zdd)
		xlast = x
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in x:
			for nbr in W[n]:
				x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
			x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
		err = sum([abs(x[n] - xlast[n]) for n in x])
		ydd.append(err)
		if err < N*tol:
			xdd1 = np.array(xdd)
			ydd1 = np.array(ydd)
			plt.plot(xdd1, ydd1)
			fname = './test.pdf'
			plt.savefig(fname)
			proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
			return 0
			
def update(r):
    res_nodes, res_values, it = r
    res_values = np.asarray(res_values).ravel()
    plt_nodes = nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        nodelist=res_nodes,
        node_color=res_values,
        alpha=1,
        node_size=700,
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=0.2
    )
    ax.axis("off")
    ax.set_title(f"Iteration {it}")
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=14)
    return [plt_nodes, ]



            
def main():
    create_files()
    G = nx.read_edgelist('./waga.txt',nodetype=int,
  	data=(('weight',float),), create_using=nx.DiGraph())
    print(G.edges(data=True))
    nx.draw(G)
    plt.show()
    
    G = nx.read_edgelist('./zbior.txt',nodetype=int,
  	data=(('weight',float),), create_using=nx.DiGraph())
    pr = nx.pagerank(G,0.4)
    #print(pr)
    
    pagerank()
    
    pagerank_chart()

if __name__ == "__main__":
    main()

