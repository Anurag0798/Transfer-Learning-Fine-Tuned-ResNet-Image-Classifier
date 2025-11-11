# Transfer Learning Fine Tuned ResNet Image Classifier

## Overview
This project implements a transfer-learning workflow that fine-tunes a ResNet backbone (Hugging Face / TFResNetModel) for a custom image classification task using TensorFlow/Keras. It includes data loading and preprocessing utilities, a training script that freezes the backbone and trains a small classification head, evaluation utilities, and a Streamlit-based inference app for uploading images and getting predictions. Key configuration (image size, model name, number of classes) is exposed in `config.py`.

## Highlights / Features
- Transfer learning with a Hugging Face ResNet backbone (microsoft/resnet-50) and a Keras classification head.
- Preprocessing pipeline that uses AutoImageProcessor from transformers to produce TF-compatible inputs.
- Training script saves best weights to disk (`saved_model/resnet_weights.h5`) and returns a compiled Keras model.
- Streamlit app for inference: loads the trained model and processor, accepts image uploads, performs preprocessing (CHW ↔ HWC conversion), and shows predicted class + confidence score.  
- Data loading utilities that create tf.data pipelines compatible with the processor and batch/shape expectations.
- Docker-friendly layout and example Dockerfile pattern available for containerized serving (see Dockerfile).

## Repo layout
- `config.py` - central config: IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS, MODEL_NAME (e.g., "microsoft/resnet-50").  
- `data_loader.py` - image preprocessing and dataset builder that uses AutoImageProcessor and wraps preprocessing with tf.py_function for use in tf.data pipelines.
- `train.py` - builds the transfer-learning model (TFResNetModel backbone + dense head), compiles and trains it, and saves weights to `saved_model/resnet_weights.h5`.
- `main.py` - example entrypoint that loads datasets, calls training, runs evaluation, and persists the final model (example saves model to `saved_model/ResNet_Image_Classifier`).
- `evaluate.py` - small evaluation helper to print validation accuracy.
- `app.py` - Streamlit UI for inference: loads the fine-tuned model and processor, preprocesses uploaded images and displays predicted class + confidence.
- `requirements.txt` - lists required packages (tensorflow, transformers, datasets, huggingface_hub, etc.) - ensure you install these versions for compatibility.
- `Dockerfile` - example container image pattern to run the Streamlit app (`streamlit run app.py`) on port 8501.
- `structure_builder.py` - helper to scaffold the project structure (optional).

## Requirements
Install Python 3.8+ and required Python packages. The repository includes a requirements file that recommends specific libraries (TensorFlow and Hugging Face stack):

Example:
```bash
pip install -r requirements.txt
```
Requirements include (see `requirements.txt`):
- tensorflow (>= 2.11.0)
- transformers (>= 4.36.0)
- datasets, huggingface_hub
- pillow, numpy, scikit-learn, matplotlib
- streamlit (for the app)
Ensure versions are compatible with the saved model and your TF installation.

## Prepare data
- Place your training and validation datasets under folders expected by the loader (example layout used by the repo):
  ```
  dataset/
    train/
      class_1/
      class_2/
      ...
    val/
      class_1/
      class_2/
      ...
  ```
- `data_loader.py` expects images and labels as inputs and will build tf.data pipelines that:
  - use AutoImageProcessor.from_pretrained(MODEL_NAME) to preprocess images,
  - convert the processor CHW format -> HWC for TensorFlow,
  - return batched datasets ready for training/evaluation.

## Training (quickstart)
1. Edit `config.py` as needed (IMAGE_SIZE, NUM_CLASSES, MODEL_NAME).
2. Prepare datasets as above.  
3. Run training:
```bash
python train.py
```
or use the example pipeline in `main.py` which calls the training function and saves the final model.
Training will:
- load the TFResNetModel backbone and freeze it for transfer learning,
- add a small dense head,
- compile with Adam and categorical crossentropy,
- run for EPOCHS (from config) and save weights to `saved_model/resnet_weights.h5`.

Notes:
- `train.py` saves weights with `model.save_weights("saved_model/resnet_weights.h5")` - adjust paths if you prefer saving the whole model.
- `main.py` demonstrates saving the full model (using model.save) and running evaluation.

## Evaluation
After training, evaluate using the provided evaluate helper:
```bash
python -c "from evaluate import evaluate_model; evaluate_model(model, val_ds)"
```
Or run the `main.py` pipeline which calls training → evaluate → save flow.

## Inference / Streamlit app
- Start the app:
```bash
streamlit run app.py
```
- The app:
  - loads the model (ensure it points to the model/weights you saved),
  - loads the AutoImageProcessor (used in training),
  - preprocesses an uploaded image, transposes/reshapes to model input, and runs `model.predict()` to obtain class probabilities and predicted label.
- `app.py` includes UI rendering of predicted class and confidence score as well as optional display of all class probabilities.

## Configuration and tips
- MODEL_NAME is set in `config.py` (default: "microsoft/resnet-50") - changing this will change the processor/backbone used; ensure the IMAGE_SIZE and preprocessing match the model.  
- Keep preprocessing identical between training and inference: the repo uses the Hugging Face processor to ensure consistent pixel scaling and normalization.
- When saving/loading models, prefer storing both the processor and model weights/artifacts so inference can be reproduced exactly.

## Common issues & troubleshooting
- Import / version errors: ensure `transformers` and `tensorflow` versions are compatible (TF >= 2.11 recommended).
- Artifact mismatch: if inference fails with shape errors, verify `IMAGE_SIZE` and the transpose operations (CHW ↔ HWC) used in `data_loader.py` and `app.py`.
- If GPU training is desired, run on a CUDA-enabled environment with appropriate TF build.

## Reproducibility
- Pin package versions in `requirements.txt`.  
- Save both the model weights and the processor configuration (e.g., using `AutoImageProcessor.save_pretrained()` or saving the model name used) so inference can reconstruct preprocessing exactly.

## Contributing
Contributions welcome. Suggested workflow:
1. Fork the repo.  
2. Create a branch and implement changes.  
3. Add tests / small examples, update README and config as needed.  
4. Open a PR.

## License
LICENSE file added. Please to it for further details.