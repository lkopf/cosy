# GloVal: Validation of Global Explanations

Evaluation method for global explanations

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
