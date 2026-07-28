# Phase 5 — Package & Publish

> **Goal:** Make PH-Neuro usable by others. Release package, paper, and pre-trained models.  
> **Duration:** ~2-3 weeks  
> **Success:** `pip install ph-neuro` works, paper submitted to conference

---

## 5.1 Package

### API Design

```python
# Minimal API — one import, one line to train
from ph_neuro import HebbianModel, HebbianTrainer

model = HebbianModel([
    HebbianModel.linear(784, 256, theta_upper=5.0),
    HebbianModel.activation('sign'),
    HebbianModel.linear(256, 10, theta_upper=5.0),
])

trainer = HebbianTrainer(
    model,
    lr=0.01,
    decay=1e-5,
    hebbian_rule='basic',  # 'basic', 'bcm', 'oja', 'anti_hebbian'
)

trainer.fit(train_loader, epochs=10)

# Access weights
model.layers[0].weights  # TernaryTensor
model.layers[0].latent_scores  # torch.Tensor (fp16)

# Save/load
model.save('mnist_model.phn')
model = HebbianModel.load('mnist_model.phn')
```

### Package Structure

```
ph_neuro/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── ternary_tensor.py      # TernaryTensor (2-bit packed storage)
│   ├── latent_scores.py        # LatentScoreTensor (fp16 scores)
│   └── hebbian_rules.py        # Basic, BCM, Oja, Anti-Hebbian
├── layers/
│   ├── __init__.py
│   ├── linear.py               # TernaryHebbianLinear
│   ├── conv.py                 # TernaryHebbianConv2d
│   ├── embedding.py            # TernaryHebbianEmbedding
│   └── attention.py            # TernaryHebbianAttention
├── models/
│   ├── __init__.py
│   ├── mlp.py                  # Hebbian MLP builder
│   ├── cnn.py                  # Hebbian CNN builder
│   └── transformer.py          # Hebbian Transformer builder
├── training/
│   ├── __init__.py
│   ├── trainer.py              # HebbianTrainer
│   ├── callbacks.py            # Logging, checkpointing, early stopping
│   └── data.py                 # Data loading utilities
├── analysis/
│   ├── __init__.py
│   ├── visualization.py        # Weight histograms, synapse tracking
│   └── continual.py            # Continual learning evaluation
├── utils/
│   ├── __init__.py
│   ├── popcount.py             # Popcount MatMul (CUDA extension)
│   └── packing.py              # Weight packing/unpacking utilities
└── examples/
    ├── mnist_mlp.py
    ├── cifar10_cnn.py
    ├── split_mnist_continual.py
    └── tinystories_transformer.py
```

### PyPI Release

- [ ] `pyproject.toml` with dependencies (torch >= 2.0)
- [ ] Version: 0.1.0 (pre-alpha)
- [ ] `pip install ph-neuro`
- [ ] CI/CD: GitHub Actions for testing on push
- [ ] Documentation: mkdocs or sphinx, hosted on GitHub Pages

### Pre-trained Model Zoo

| Model | Task | Accuracy | Download |
|-------|------|----------|----------|
| ph-neuro-mnist-mlp | MNIST | >95% | `HebbianModel.from_pretrained('ph-neuro/mnist-mlp')` |
| ph-neuro-cifar10-cnn | CIFAR-10 | >60% | ... |
| ph-neuro-tinystories-100m | TinyStories generation | — | ... |
| ph-neuro-1b | Language model | — | ... |

---

## 5.2 Paper

### Title (Draft)

> **Ternary Hebbian Networks: Backprop-Free Deep Learning with Discrete Synaptic Weights**

### Abstract (Draft)

> We introduce PH-Neuro, a deep learning framework that combines ternary weights {-1, 0, +1} with Hebbian plasticity, eliminating backpropagation entirely. Each synapse updates based solely on the local activity of its pre- and post-synaptic neurons (ΔW = pre × post), requiring no global loss function, no gradient computation, and no optimizer states. We demonstrate that ternary Hebbian networks achieve competitive performance on MNIST (>95%) and CIFAR-10 (XX%), while using ~50× less training memory and ~6.5× fewer FLOPs than equivalent backprop-trained networks. Crucially, ternary Hebbian networks exhibit negligible catastrophic forgetting (<5% on split MNIST vs >40% for backprop), enabling genuine continual learning without replay buffers. We further present the first Hebbian-trained language model (100M parameters on TinyStories), which generates coherent text despite never computing a gradient. Our results suggest that ternary Hebbian learning occupies a unique Pareto-optimal point in the accuracy-efficiency-continual learning trade space, opening new directions for edge deployment, online adaptation, and biologically plausible AI.

### Key Contributions

1. **First combination of ternary weights with Hebbian learning** (genuinely novel)
2. **Hysteresis threshold mechanism** for stable ternary weight transitions
3. **Demonstration that ternary Hebbian eliminates catastrophic forgetting** while maintaining useful accuracy
4. **First Hebbian language model** (even if performance is modest)
5. **Memory and compute analysis** showing 50× memory reduction, 6.5× FLOP reduction vs backprop

### Target Venues

| Venue | Deadline | Fit |
|-------|----------|-----|
| **NeurIPS** | May (abstract), May (paper) | Best fit — welcomes alternative learning paradigms |
| **ICLR** | October | Good fit — published SoftHebb |
| **ICML** | January | Good fit — strong biological motivation |
| **TMLR** | Rolling | Backup — if we miss conference deadlines |

### Figure Plan

1. **Figure 1**: Architecture diagram — ternary weights, Hebbian update, no backward pass
2. **Figure 2**: Accuracy vs memory trade-off (Pareto frontier plot) — PH-Neuro vs backprop vs other Hebbian methods
3. **Figure 3**: Continual learning — forgetting curves for PH-Neuro vs backprop on split MNIST
4. **Figure 4**: Weight dynamics — latent score evolution, ternary weight stability, synapse lifetime distribution
5. **Figure 5**: Language model samples — generated text from TinyStories model
6. **Figure 6**: Scaling behavior — perplexity vs model size for Hebbian vs backprop

---

## 5.3 Community & Outreach

- [ ] **Blog post**: "Learning Without Gradients: Introducing PH-Neuro" — accessible intro
- [ ] **Twitter thread**: Key results in visual form
- [ ] **Colab notebook**: Train a Hebbian MNIST classifier in 5 minutes
- [ ] **Demo**: Interactive weight visualization — watch synapses form in real time
- [ ] **Hacker News / Reddit**: Share the paper and code
- [ ] **Video**: 5-minute explainer with training visualization

---

## 5.4 Future Horizons (Phase 6+)

These are beyond the initial roadmap but worth sketching:

### Neuromorphic Hardware
- Intel Loihi 2: native support for ternary weights and local learning rules
- SpiNNaker: spiking neural network hardware
- PH-Neuro's design maps naturally to neuromorphic chips

### WebAssembly Inference
- Compile ternary MatMul to WASM with SIMD
- Run Hebbian inference in the browser
- Demo: in-browser continual learning (learns from user interactions)

### Federated Hebbian Learning
- Edge devices train locally with Hebbian updates
- Merge ternary weights by majority vote (no gradient averaging needed!)
- Privacy-preserving: only discrete weights are shared, not data

### Hebbian Fine-Tuning
- Take a pre-trained (backprop) model
- Quantize to ternary
- Fine-tune with Hebbian learning on new data
- Adapt to new domains without catastrophic forgetting

### Liquid PH-Neuro
- Continuous-time weight dynamics (dW/dt instead of ΔW per step)
- Differential equations governing synaptic plasticity
- More brain-like: constant adaptation, no discrete "epochs"

---

## Deliverables

- [ ] `ph-neuro` pip package published on PyPI
- [ ] Documentation site with quickstart, API reference, examples
- [ ] Pre-trained models on HuggingFace Hub
- [ ] ArXiv preprint
- [ ] Conference submission
- [ ] Blog post + social media
- [ ] Colab demo notebook

---

> *PH-Neuro is not the end of backprop. It's the beginning of an alternative — one that learns like nature does, one synapse at a time.*
