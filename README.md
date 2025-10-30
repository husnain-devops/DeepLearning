# DeepLearning

A small collection of deep learning notebooks showcasing:

- Transfer learning on the PlantVillage dataset with InceptionV3
- Transfer learning on the PlantVillage dataset with ResNet50
- A from-scratch 3-layer DNN in PyTorch with manual backprop and Adam update

## Notebooks

- `TransferLearning-InceptionV3.ipynb`
  - TensorFlow/Keras implementation using InceptionV3 (ImageNet weights)
  - 80/10/10 train/val/test split via `tensorflow_datasets` PlantVillage
  - CPU-only execution configured; seed set for reproducibility
  - Saves artifacts: `best_inceptionv3_model.h5`, `inceptionv3_training_history.png`, `inceptionv3_confusion_matrix.png`, `inceptionv3_results.csv`
  - Reported results:
    - Accuracy: 95.36%
    - Precision: 95.34%
    - Recall: 95.36%
    - F1-Score: 95.32%

- `TransferLearning-ResNet50.ipynb`
  - TensorFlow/Keras implementation using ResNet50 (ImageNet weights)
  - Partial fine-tuning (top ~15% of layers unfrozen)
  - 80/10/10 train/val/test split via `tensorflow_datasets` PlantVillage
  - CPU-only execution configured; seed set for reproducibility
  - Saves artifacts: `best_resnet50_model.h5`, `resnet50_training_history.png`, `resnet50_confusion_matrix.png`, `resnet50_results.csv`
  - Reported results:
    - Accuracy: 63.19%
    - Precision: 74.26%
    - Recall: 63.19%
    - F1-Score: 62.68%

- `3L-DNN.ipynb`
  - PyTorch toy example of a 3-layer DNN
  - Manual forward pass, gradient derivation, and a single Adam update step implemented from first principles
  - Demonstrates building computational graph, computing deltas for tanh/ReLU/sigmoid layers, and applying Adam with moment estimates

## Environment & Dependencies

Recommended: Python 3.10+

Core libraries used across notebooks:

- TensorFlow 2.x, Keras
- TensorFlow Datasets (`tensorflow-datasets`)
- PyTorch (`torch`)
- NumPy, Pandas
- scikit-learn (metrics)
- Matplotlib, Seaborn

Install (CPU-only example):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install tensorflow==2.14.* tensorflow-datasets torch torchvision torchaudio \
            numpy pandas scikit-learn matplotlib seaborn
```

Note: TensorFlow and PyTorch wheels vary by OS/CPU/GPU. If you have a GPU, install the appropriate GPU-enabled builds from the official docs.

## Dataset

Both transfer learning notebooks use PlantVillage via `tensorflow_datasets`:

- Name: `plant_village`
- Split: 80% train, 10% validation, 10% test

The dataset will be downloaded automatically on first run.

## How to Run

1. Create and activate a virtual environment and install dependencies (see above).
2. Launch Jupyter:
   ```bash
   pip install notebook
   jupyter notebook
   ```
3. Open one of the notebooks and run all cells.

Artifacts (models, plots, CSVs) are written to the project root by default.

## Results Summary

| Notebook                         | Accuracy | Precision | Recall | F1-Score |
|----------------------------------|---------:|----------:|-------:|---------:|
| TransferLearning-InceptionV3     |   95.36% |    95.34% | 95.36% |   95.32% |
| TransferLearning-ResNet50        |   63.19% |    74.26% | 63.19% |   62.68% |
| 3L-DNN (toy, PyTorch)            |     N/A  |       N/A |    N/A |      N/A |

`3L-DNN.ipynb` is an educational example (no dataset benchmark), illustrating manual backprop and Adam updates.

## Notes

- Both TF notebooks force CPU execution; remove the device restriction if you want to use a GPU.
- Seeds are set for reproducibility, though nondeterminism may still exist across platforms/backends.