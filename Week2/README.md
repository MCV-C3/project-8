## Project Overview
This project implements a baseline neural network model using PyTorch for classification tasks. It trains a simple feedforward neural network on the **MIT Indoor Scenes dataset** and visualizes the training progress through metrics like loss and accuracy. The project includes functionality to visualize the computational graph of the model.

## Requirements
- Python 3.8+
- Required Python libraries:
  - PyTorch (`torch`, `torchvision`)
  - Numpy (`numpy`)
  - Matplotlib (`matplotlib`)
  - Torchviz (`torchviz`)
  - Tqdm (`tqdm`)

To install the required packages check the cuda version and install the torch version that fits in it

## Code Structure
1. **`main.py`**: Main script to train and evaluate the model. It also handles data loading and visualization of metrics.
2. **`models.py`**: Contains the definition of the `SimpleModel` class, a basic feedforward neural network with customizable input, hidden, and output dimensions.

## Usage

### Dataset Preparation
The dataset should follow this structure:
```
~/data/Master/MIT_split/
  ├── train/
  │   ├── <class_label>/
  │   │   ├── img1.jpg
  │   │   ├── img2.jpg
  ├── test/
      ├── <class_label>/
          ├── img1.jpg
```
Replace `<class_label>` with appropriate labels (e.g., "bedroom," "office").

### Run the Script
To execute the baseline model:
```bash
python main.py
```

### Training and Testing
1. The `train` function optimizes the model using the Adam optimizer and CrossEntropy loss.
2. The `test` function evaluates the model on the test set and computes the loss and accuracy.

### Visualizations
- **Metric Plots**: Training and test loss/accuracy over epochs are saved as `loss.png` and `metrics.png`.
- **Computational Graph**: The model's computational graph is saved as `computational_graph.png`.

## Customization

### Model Customization
To modify the architecture, adjust the parameters in the `SimpleModel` initialization:
```python
model = SimpleModel(input_d=(C × H × W), hidden_d=300, output_d=8)
```
- `input_d`: Flattened size of input image (C × H × W).
- `hidden_d`: Number of hidden units in each layer.
- `output_d`: Number of output classes.

### Data Loading
Change the dataset path or batch sizes in the `DataLoader` definitions:
```python
train_loader = DataLoader(data_train, batch_size=256, num_workers=8)
```

### Training Configuration
- Adjust the number of epochs:
  ```python
  num_epochs = 20
  ```
- Modify learning rate or optimizer:
  ```python
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  ```

## Results
The script outputs the training and test loss/accuracy for each epoch. Metrics are visualized and saved as plots.
