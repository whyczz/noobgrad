
## Changelog

- [x] 0.1 Add a Scalar with a output, repr
- [x] 0.2 Add operations (add, mul, exp) to Scalar
- [x] 0.3 Reflected operators, convenience unwrapping ints or floats
- [x] 0.4 Trace the history of operators, graph it
- [x] 0.5 Add backwards, gradient accumulation e.g. sums
- [x] 0.5.1 Add tanh() and exp() forwards, backwards
- [x] 0.6 Neural Net
    - [x] 0.6.1 Neuron
    - [x] 0.6.2 Layer
    - [x] 0.6.3 MLP
- [x] 0.7 Training loop
    - [x] 0.7.1 loss function - mean squared error
    - [x] 0.7.2 single loop - forwards, backwards, gradient descent
- [ ] 0.8 Training epoch
    - [ ] 0.8.1 Run loop for N iterations
    - [ ] 0.8.2 Print loss each epoch, observe decrease, graph it
    - [ ] 0.8.3 Test predictions after training (should be close to targets)
    - [ ] 0.8.4 Graph model boundaries for 2D