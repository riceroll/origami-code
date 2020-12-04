$x$ :	current node positions	```x: [n_nodes, 3]```

$x_{init}$ :	initial node positions	```x_0: [n_nodes, 3]```

$e$:	edges, storing indices of two nodes	```e: [n_bars = n_edges, 2]```

$f$:	faces, storing indices of three nodes in anti-clock wise order	```face: [n_faces, 3]```



#### 1. bar forces

$$
{{f_{bar}}} = -M_{bar}^\intercal {f_{bar}}_{i}    \\
{f_{bar}}_{i} = \frac{EA}{L_0} \frac{\|d\|_2-L_0}{\|d\|_2}d \\
d_{init} = M_{bar} x_{init}\\
L_0 = \|d_{init}\|_2\\
d = M_{bar} x\\
$$

$f_{bar}$: bar forces applied on all nodes	```f_bar: [n_nodes, 3]```

${f_{bar}}_{i}$:	bar internal force applied on node $x_i$ ```f_bar_in: [n_bars, 3]```

$M_{bar}$:	bar matrix	```M_bar: [n_bars, n_nodes]``` 

$d$:	displacement $x_j - x_i$, $d_{init}$ is initial displacement	```d: [n_bars, 3]```

$L_0$:	original length of bars	```L_0: [n_bars, 1]```



#### 2. hinge forces

$$
f_{hinge} = -k_{hinge} (\rho - \rho_{target})\frac{\partial \rho}{\partial x}\\
\frac{\partial \rho}{\partial x_k} = \frac{n_{k}}{h_{k}}\\
\frac{\partial \rho}{x_i} = \frac{-\cot{\theta_{jik}}}{\cot{\theta_{jik}} + \cot{\theta_{ijk}}}
								\frac{n_{k}}{h_{k}} +
								\frac{-\cot_{\theta_{jil}}}{\cot{\theta_{jil}} + \cot{\theta_{ijl}}}
                \frac{n_{l}}{h_{l}}	\\
$$

(refer to **Fig. 4** & **Eq. 15-18** from *Freeform Variations of Origami*)
$$

$$

$$
n_{k} = \frac{d_{ij} \times d_{jk}}{\| d_{ij} \times d_{jk} \|_2} \\
h_{k} = \frac{\|d_{ij} \times d_{jk}\|_2}{\|d_{ij}\|_2}\\
\cot{\theta_{jik}} = \frac{\cos{\theta_{jik}}}{\sqrt{1 - \cos{\theta_{jik}^2}}}\\
\cos{\theta_{jik}} = \frac{\|d_{ij}\|_2^2 + \|d_{jk}\|_2^2 - \|d_{ik}\|_2^2 }{2 \|d_{ij}\|_2 \|d_{jk}\|_2}\\
x_i = {M_{hinge}}_i x	\\
d_{ij} = x_j - x_i	\\
$$
$x_i, x_j, x_k, x_l$:	nodes positions of a hinge	```x_i: [n_hinges x 3]```

${M_{hinge}}_i,{M_{hinge}}_j,{M_{hinge}}_k,{M_{hinge}}_l$:	hinge matrix	```M_hinge_i: [n_hinges, n_nodes]```

$d_{ij}, d_{ik}, d_{jk}, d_{il}, d_{jl}$: edge vector$x_j - x_i$	```d_ij: [n_hinges, 3]```

$n_{k}, n_{l}$:	unit normal vectors of the two faces	```n_k: [n_hinges, 3]```

$h_{k}, h_{l}$:	height from two end nodes to the hinge	```h_k: [n_hinges, 1]```



#### 3. face forces

$$
{f_{face}}_{ji} = -k_{face} (\theta_j - {\theta_{j}}_0) \frac{\partial \theta_j}{\partial x_i} \\
\frac{\partial \theta_{j}}{\partial x_i} = \frac{n \times (x_i - x_j)}{\|x_i - x_j\|_2^2}\\
\frac{\partial \theta_{j}}{\partial x_j}=\frac{n \times (x_j - x_i)}{\|x_i - x_j\|_2^2} + 
\frac{n \times (x_k - x_j)}{\| x_k - x_j \|_2^2} \\
$$

(refer to **Fig. 4** & **Eq. 12-14** from *Freeform Variations of Origami*)

${\theta_j}_0$: 	 initial angle of $\theta$	



#### Optimization for Kawazaki Theorem

$$
\theta_l = \arccos \frac{(x_u - x) \cdot (x_l - x)}{\|(x_u - x)\| \cdot \| (x_l - x)\|}\\
\theta_r = \arccos \frac{(x_d - x) \cdot (x_r - x)}{\|(x_d - x)\| \cdot \| (x_r - x)\|}\\
\arg\min_{x} (\theta_l + \theta_r - \pi)^2
$$

$x$: node positions on 2D, which are inner nodes

$x_u, x_d, x_l, x_r $: upper/down/left/right node positions

