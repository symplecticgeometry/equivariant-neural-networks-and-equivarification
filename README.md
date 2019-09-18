# Equivariant neural networks and equivarification
This is the implementation of an equivariant neural network on the MNIST data set.
## What is an equivariant neural network?
Let *M* be the space of data (e.g. the space of all rgb pictures of the size 225 * 225 * 3).
Let *N* be the space of all possible configuration of classes. e.g. 
*N = { (animal, angle) | animal = cat or dog, angle = 0, pi/2, pi, 3pi/2}*.

Then a neural network is a function *f* from *M* to *N*.

Let *G* be a group that acts on *M*. 
E.g. *G* has four elements, and denote it by *G = {r0,r1,r2,r3}*. And *ri* acts on image in *M* by rotating it by *i \* pi/2* in the counterclockwise direction. 

We also require that *G** acts on *N* (e.g.  *rk (animal, angle) = (animal, angle + k\*pi/2 (mod 2 pi)) for k = 0,1,2,3*.

**We say the neural network *f* is *G*-equivariant, if**
```
f(g m) = g f(m), 
```
**for all *m* in *M* and *g* in *G*.**
## How to construct an equivariant neural network
The idea of achieving an equivariant neural network is the following (the [paper](https://arxiv.org/abs/1906.07172) deals  with more general cases, i.e. an arbitrary group *G*):

Given function *f: M -> N* as above.

We modify *f* and *N* to: 
*F: M -> N'*, so that *N'* is an "enlargement" of *N*, *G* acts on *N'*, and *F* is *G*-equivariant as follows.
Define *N' = N^4 = Y cross Y cross Y cross Y*. (here we assume G is a cyclic group of order 4, please refer to the [paper](https://arxiv.org/abs/1906.07172) for general case).
H(x) = (h(x), h(r1 x), h(r2 x), h(r3 x)).
To define a G action on Z, we only need to define how r1 acts on Z:
 r1 (y0, y1, y2, y3) = (y1, y2, y3, y0).

Suppose h:X->Y is the first layer of a cnn, then we modify it into H:X->Z,
and Z now has a G-action. So we build another standard cnn layer starting from Z
and then using the same technic we make it equivariant. Alternatively, a less interesting thing that one can do is to use the original next level layer from Y, and precompose it with the projection p from Z to Y. Here p: Z to Y is given by 
p(y0, y1, y2, y3) = y0.
Either way, inductively, we get an equivariant neural network.

--to do: clean up the code.
