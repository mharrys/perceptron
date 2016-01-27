Perceptron
==========
Small experiments with artificial neural networks in Torch7.

One Layer Perceptron
--------------------
Learn with the delta rule how to classify linear seperable data, the script will
animate the separation line during training.

![fig1](https://github.com/mharrys/perceptron/raw/master/img/one_layer_perceptron_result.png)

Multi-Layer Perceptron (Two Layers)
-----------------------------------
Learn with backprop how to classify non-linaer seperable data.

![fig2](https://github.com/mharrys/perceptron/raw/master/img/two_layer_perceptron_points.png)
![fig3](https://github.com/mharrys/perceptron/raw/master/img/two_layer_perceptron_result.png)

Encoder
-------
A classical encoder problem where the network is forced to find a compact
coding of sparse data. The network has a hour glass shaped topology (8-3-8).

Function Approximation
----------------------
Learn with backprop how to approximate a function (Gauss), the first plot show
the exact function values and the second shows the approximation. Experiment
by adjusting the number of hidden nodes and watch how the network will
underfit or overfit.

![fig4](https://github.com/mharrys/perceptron/raw/master/img/approx_actual.png)
![fig5](https://github.com/mharrys/perceptron/raw/master/img/approx_expected.png)

References
==========
KTH notes.
