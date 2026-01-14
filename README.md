
## Changelog

- [0.1] Add a Scalar with a output, repr
- [0.2] Add operations (add, mul, exp) to Scalar
- [0.3] Reflected operators, convenience unwrapping ints or floats
- [0.4] Trace the history of operators, graph it
- [0.5] Add backwards, gradient accumulation e.g. sums
- [0.5.1] Add tanh() and exp() forwards, backwards 
- [0.6] Neural Net
- [0.6.1] Neuron
- [0.6.2] Layer
- [0.6.3] MLP
- [0.7] Training loop
    - [x] 0.7.1 loss function - mean squared average
    - [x] 0.7.2 single loop - e.g `s.data += learning_rate*s.grad`
- [0.8] Training epoch 