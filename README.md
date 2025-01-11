# Flower Classification with ResNet50

This project demonstrates how to classify flower images into 102 categories using a pre-trained ResNet50 model fine-tuned on a custom flower dataset.

## Project Workflow

1. **Data Preparation**:
   - Flower images and labels are loaded from the dataset.
   - Dataset splits: Training, Validation, and Testing.
   - Images are preprocessed using transformations compatible with ResNet50.

2. **Custom Dataset Class**:
   - A PyTorch `Dataset` class is created to handle image loading and label mapping.

3. **Model Setup**:
   - ResNet50 is fine-tuned with its final fully connected layer modified to classify 102 flower species.

4. **Training**:
   - The model is trained using the CrossEntropy loss and Adam optimizer.

5. **Evaluation**:
   - Test accuracy is computed on unseen test data.

## Results

- Achieved test accuracy: **0.841645**
- Achieved validation accuracy: **0.8574**


## Project Structure

```
102-flowers-ResNet50/
├── data/                   # Dataset folder
├── train.py                # Training, Evaluation and Testing script
├── models/                 # Saved model weights
├── model_training.ipynb    # Project Notebook        
├── utils.py                # Data (image) visualization script        
├── README.md               # Project README
     
```

## Acknowledgments

- Dataset sourced from the Oxford 102 Flower Dataset. https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Model architecture: ResNet50 from torchvision.

---

Learning Project by [Dipesh Pandit](https://github.com/dipesh1dp).
