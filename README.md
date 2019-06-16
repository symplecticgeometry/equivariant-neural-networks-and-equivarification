This is the implementation of an equivariant neural network using the MNIST data set.

Let M be the space of data (e.g. the space of all rgb pictures of the size 225 * 225 * 3).
Let N be the space of all possible configuration (e.g. N = { (animal, angle) | animal = cat or dog, angle = 0, pi/2, pi, 3pi/2})

Then a neural network is a function f from M to N.
Let G be a group that acts on M. 
(e.g. G is the group of rotation by pi/2 in the counterclockwise direction. 
In this case, denote G = {r0,r1,r2,r3}, the way that G acts on M is given by the natural rotation)

We also require that G acts on N (e.g.  rk (animal, angle) = (animal, angle + kpi/2 (mod 2 pi)) for k = 0,1,2,3.

We say the neural network f is G-equivariant, if f(g m) = g f(m), for all m in M and g in G.

The idea of achieving an equivariant neural network is the following:

given function h: X -> Y, where X, Y are two spaces, and G acts on X.

We now modify h and Y to: 
H: X -> Z, so that G acts on Z, and H is G-equivariant.
Z is constructed by Y^4 = Y cross Y cross Y cross Y. (here we assume G is a cyclic group of order 4, please refer to the paper (equivariant neural network and equivarification) for general case).

H(x) = (h(x), h(r1 x), h(r2 x), h(r3 x)).
To define a G action on Z, we only need to define how r1 acts on Z:
 r1 (y1, y2, y3, y4) = (y2, y3, y4, y1).

suppose h:X->Y is the first layer of a cnn, then we modify it into H:X->Z,
and Z now has a G-action. So we build another standard cnn layer starting from Z
and then using the same technic we make it equivariant. Inductively, we get an equivariant neural network.
