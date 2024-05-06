# Concept Synthesis: Evaluation of Concept-based Textual Explanations

Evaluation method for concept-based textual explanations

## 0. Collect Explanations with Explanation Methods

TODO: Write instructions on how to set up the following repos:

- [MILAN](https://github.com/evandez/neuron-descriptions)
- [FALCON](https://github.com/NehaKalibhat/falcon-explain)
- [CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)
- [INVERT](https://github.com/lapalap/invert)

## 1. Collect Activations

Collect activations for your model

```bash
python src/activation_collector.py
```

## 2. Generate Explanation Images

```bash
torchrun src/image_generator.py --nproc_per_node=3
```

## 3. Evaluate Explanations

```bash
python src/evaluation.py
```
