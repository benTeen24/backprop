---
title: Backprop
---

# Welcome to Backprop

Notes on papers and ideas I'm reading in AI/ML.

## Sample note: backpropagation

The weight update rule during gradient descent:

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w} 
$$

where $\eta$ is the learning rate and $L$ is the loss. In code, one SGD step looks like:

```python
def sgd_step(w, grad, lr=0.01):
    return w - lr * grad
```

Reading list:

- [x] Attention Is All You Need
- [ ] Direct Preference Optimization (DPO)
- [ ] Mamba: Linear-Time Sequence Modeling

See [[Welcome]] for the default Obsidian note, or check the graph view in the sidebar.
