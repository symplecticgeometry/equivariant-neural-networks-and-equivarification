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
## How to construct an equivariant neural network?
The idea of achieving an equivariant neural network is the following (the [paper](https://arxiv.org/abs/1906.07172) deals  with more general cases, i.e. an arbitrary group *G*):

Given function *f: M -> N* as above.

We modify *f* and *N* to: 
*F: M -> N'*, so that *N'* is an "enlargement" of *N*, *G* acts on *N'*, and *F* is *G*-equivariant as follows.
Define *N' = N^4 = N *x* N *x* N *x* N*. (Here we assume as before *G* is a cyclic group of order 4, please refer to the [paper](https://arxiv.org/abs/1906.07172) for the general case).
We define 
```
F(m) = (h(m), h(r1 m), h(r2 m), h(r3 m)),
```
where *rk m* means rotating the image *m* counterclockwise by *k\*pi/2*.
To define a *G* action on *N'*, we only need to define how *r1* acts on *N:
 ```
 r1 (n0, n1, n2, n3) = (n1, n2, n3, n0).
 ```

Suppose *f:M->N is the first layer of a cnn, then we modify it into *F:M->N'*,
and *N'* now has a *G*-action. So we build another standard cnn layer starting from *N'*
and then using the same technic we make it equivariant. Alternatively, a less interesting thing that one can do is to use the original next level layer from *N*, and precompose it with the projection p from *N'* to *N*. Here *p: N' -> N* is given by 
```
p(n0, n1, n2, n3) = n0.
```
Either way, inductively, we get an equivariant neural network.

## Implementation and versions
The code is written in tensorflow. There are two versions. V2 is a better version, and it can handle arbitrary neural networks and arbitrary groups. (V1 is not overwritten due to paper submission requirements).

--to do: clean up the code.
