# eXtensible Variational Inference

Bayesian methods allow us to capture uncertainty while drawing inferences and making predictions from
data. Instead of point estimates, they provide the machinery to generate a probability distribution
over the inference targets.

This library provides a scalable and extensible framework for Bayesian Inference using the mean field 
simplification of [Auto Differentiation Variational Inference](https://jmlr.org/papers/v18/16-107.html) 
developed by developed by Kucukelbir at al. 

It is implemented on top of TensorFlow and can be easily integrated with the rest of its ecosystem. 
