# Hi, I'm Qui Pham

Computer Science and Engineering student at **UC Irvine**, working on machine learning research:
optimal transport algorithms and generative models.

- **ML researcher at UC Irvine** with Prof. Alex Berg: the expressive power of deep networks, and
  physics-consistent video generation with flow matching.
- **Optimal transport researcher** with mentors at MIT and Georgia Tech: accelerated Sinkhorn-type
  algorithms for (partial) optimal transport.
- I like measuring things carefully: checking against a reference implementation, running enough
  seeds to tell an effect from noise, and reporting null results.

## Projects

### [Stable Diffusion](https://github.com/qpp112/Stable-Diffusion)
Stable Diffusion 1.5 and SDXL in plain PyTorch, without `diffusers` or `transformers` at runtime:
UNet, VAE, both CLIP text encoders, the BPE tokenizer, and DDPM / DDIM / DPM-Solver++ samplers. The
official weights load into these modules.
- Tokenizer, text encoders and SD 1.5 UNet match Hugging Face `diffusers` **bit for bit**. The rest
  matches to within fp16 round-off.
- SDXL at 1024×1024 in **5.2 GiB** of GPU memory, down from 16.6 GiB. It still runs with the allocator
  capped at 8 GiB. Uses fp16, fused attention, tiled VAE decoding and model offloading, all benchmarked
  on UCI's HPC3 cluster.

### [Do a Mixture-of-Experts head and class-balanced losses help toxic comment classification?](https://github.com/qpp112/toxic-comment-moe)
A course project rebuilt as a controlled study: 50+ BERT-base runs on Kaggle's Jigsaw data (160k
comments), 3 seeds each, compared pairwise by seed.
- **Neither helps.** Every head and loss ties plain BERT + BCE (test ROC-AUC 0.9855 ± 0.0001).
- **Why:** rare labels co-occur with `toxic` 93–100% of the time, so the MoE router only learns toxic
  vs. clean. Control heads (multi-gate MoE, label-wise attention, a same-size MLP) confirm the head
  isn't the bottleneck.
- **What did help** was letting BERT read more of each comment. With 512 tokens, ROC-AUC rose to
  **0.9868** on every seed, for under 10% more training time.

## Publication

**Accelerated Sinkhorn Algorithm for Partial Optimal Transport.** Nghia Truong, **Qui Phu Pham**,
Quang Nguyen, Dung Luong, Mai Tran. *ICASSP 2026.* [arXiv:2601.17196](https://arxiv.org/abs/2601.17196)

## Tools

Python · PyTorch · Hugging Face (Transformers, Diffusers) · NumPy / SciPy · scikit-learn · C++ ·
pytest · GitHub Actions · SLURM · Docker · LaTeX

## Contact

[qppham@uci.edu](mailto:qppham@uci.edu)

