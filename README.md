# Plant Disease Classifier

A Streamlit app that predicts plant diseases from leaf images using a saved TensorFlow/Keras model.

## Features

- Upload a leaf image and get a prediction.
- Shows the plant name, disease name, and a suggested solution.
- Displays the top prediction and probability ranking.
- Optional Gemini-based description generation through an environment API key.

## Project Structure

- `app.py` - Streamlit application.
- `models/plant_disease_transfer_learning.keras` - Saved model used by the app.
- `dataset/` - Training class folders used to discover label names.
- `notebooks/plant_disease.ipynb` - Notebook used for experimentation and model training.

## Requirements

Install the Python packages listed in `requirements.txt`.

## Run Locally

```bash
streamlit run app.py
```

## Optional Gemini Setup

To enable Gemini-based text explanations, create a local `.env` file from `.env.example` and set one of these environment variables before starting the app:

```bash
set GEMINI_API_KEY=your_api_key_here
```

or

```bash
set GOOGLE_API_KEY=your_api_key_here
```

The app loads `.env` locally, so your key stays on your machine and is not committed to Git.

## Notes

- The app expects images resized to `224 x 224`.
- The detected class names come from the dataset folder names.
- If the saved model does not match the training labels exactly, the result may show a generic class label.

## Publishing to GitHub

This workspace is not currently initialized as a git repository. To publish it to GitHub:

```bash
git init
git add .
git commit -m "Initial plant disease classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/plant-detection.git
git push -u origin main
```

Replace `<your-username>` with your GitHub account name.
