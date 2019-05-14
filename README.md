# Library for Linear ODE Learning (LLOL)
This library aims to generate high-dimensional linear ODE models that locally approximate the behavior of lower-dimensional nonlinear systems. This can be done in a data-driven regression approaches (extended DMD) or through symbolic techniques (Carleman linearization), or combinations of the two. This enables scalable linear systems verification and counterexample generation methods, such as those used in Hylaa (https://github.com/stanleybak/hylaa), to be applied to nonlinear systems.

# Method References
**eDMD**: Williams, Matthew O., Ioannis G. Kevrekidis, and Clarence W. Rowley. "A data–driven approximation of the koopman operator: Extending dynamic mode decomposition." Journal of Nonlinear Science 25.6 (2015): 1307-1346.

**Carleman Linearization**: Forets, Marcelo, and Amaury Pouly. "Explicit error bounds for carleman linearization." arXiv preprint arXiv:1711.02552 (2017).
