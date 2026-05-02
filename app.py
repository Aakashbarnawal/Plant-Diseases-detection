from __future__ import annotations

import json
import importlib
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple
import re
import requests

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "plant_disease_transfer_learning.keras"
DATASET_CANDIDATES = [
	ROOT / "dataset",
	ROOT / "dataset" / "PlantVillage",
	ROOT / "dataset" / "PlantVillage" / "PlantVillage",
]
EXCLUDED_FOLDER_NAMES = {"PlantVillage"}
EXPECTED_SIZE = (224, 224)

DISEASE_SOLUTIONS = {
	"Bacterial spot": "Remove infected leaves, avoid overhead watering, improve airflow, and use copper-based sprays if needed.",
	"Early blight": "Remove affected foliage, rotate crops, mulch to reduce soil splash, and apply a labeled fungicide if the disease spreads.",
	"Late blight": "Remove infected plants immediately, keep foliage dry, destroy plant debris, and use a fungicide labeled for late blight.",
	"Leaf Mold": "Increase ventilation, reduce humidity, avoid wet foliage, and remove heavily infected leaves.",
	"Septoria leaf spot": "Remove infected leaves, water at the base, rotate crops, and apply a fungicide if necessary.",
	"Spider mites Two spotted spider mite": "Wash leaves with water, use insecticidal soap or neem oil, and control dust and heat stress.",
	"Target Spot": "Remove infected debris, improve airflow, avoid overhead watering, and apply a fungicide when required.",
	"Tomato mosaic virus": "Remove infected plants, disinfect tools, control aphids, and do not reuse seeds from infected plants.",
	"Tomato yellow leaf curl virus": "Remove infected plants, control whiteflies, and use resistant varieties in the next planting cycle.",
	"healthy": "No disease detected. Keep regular watering, monitor leaves, and maintain good nutrition and airflow.",
	"Potato___healthy": "No disease detected. Keep regular watering, monitor leaves, and maintain good nutrition and airflow.",
	"Pepper__bell___healthy": "No disease detected. Keep regular watering, monitor leaves, and maintain good nutrition and airflow.",
}


def normalize_label(s: str) -> str:
	if not s:
		return ""
	# lower, remove punctuation, collapse whitespace
	import re

	t = s.lower()
	t = re.sub(r"[^a-z0-9]+", " ", t)
	t = re.sub(r"\s+", " ", t).strip()
	return t


def find_solution_for_disease(disease_name: str) -> str:
	# Try exact then fuzzy (normalized) match against DISEASE_SOLUTIONS keys
	if not disease_name:
		return ""
	if disease_name in DISEASE_SOLUTIONS:
		return DISEASE_SOLUTIONS[disease_name]
	target = normalize_label(disease_name)
	for k, v in DISEASE_SOLUTIONS.items():
		if normalize_label(k) == target:
			return v
	# fallback: substring match
	for k, v in DISEASE_SOLUTIONS.items():
		if normalize_label(k) in target or target in normalize_label(k):
			return v
	return "No specific solution was stored for this label. Use proper watering, isolate the plant, and consult an agricultural extension service for confirmation."


st.set_page_config(
	page_title="Plant Disease Classifier",
	page_icon="🌿",
	layout="wide",
)


st.markdown(
	"""
	<style>
		:root {
			--bg: #f4f1ea;
			--panel: #ffffff;
			--text: #183028;
			--muted: #5d6f68;
			--accent: #2f7d32;
			--accent-soft: rgba(47, 125, 50, 0.12);
			--border: rgba(24, 48, 40, 0.10);
		}

		.stApp {
			background:
				radial-gradient(circle at top left, rgba(47, 125, 50, 0.10), transparent 30%),
				linear-gradient(180deg, #fbfaf7 0%, #f4f1ea 100%);
			color: var(--text);
		}

		.hero {
			padding: 1.4rem 1.6rem;
			border-radius: 1.25rem;
			background: linear-gradient(135deg, #183028 0%, #2f7d32 55%, #6aa84f 100%);
			color: white;
			border: 1px solid rgba(255, 255, 255, 0.12);
			box-shadow: 0 20px 50px rgba(15, 35, 25, 0.15);
		}

		.hero h1, .hero p {
			margin: 0;
		}

		.hero p {
			opacity: 0.9;
			margin-top: 0.35rem;
		}

		.card {
			background: var(--panel);
			border: 1px solid var(--border);
			border-radius: 1rem;
			padding: 1rem 1.1rem;
			box-shadow: 0 10px 28px rgba(20, 30, 25, 0.05);
		}

		.metric-box {
			background: linear-gradient(180deg, #ffffff 0%, #f8fbf7 100%);
			border: 1px solid rgba(47, 125, 50, 0.16);
			border-radius: 1rem;
			padding: 1rem;
		}

		.small-label {
			color: var(--muted);
			font-size: 0.86rem;
			text-transform: uppercase;
			letter-spacing: 0.08em;
			margin-bottom: 0.25rem;
		}
	</style>
	""",
	unsafe_allow_html=True,
)


def resolve_label_directory() -> Path | None:
	best_path = None
	best_count = -1
	for candidate in DATASET_CANDIDATES:
		if not candidate.exists() or not candidate.is_dir():
			continue
		class_dirs = [item for item in candidate.iterdir() if item.is_dir() and not item.name.startswith(".")]
		if len(class_dirs) > best_count:
			best_path = candidate
			best_count = len(class_dirs)
	return best_path


def discover_class_names(label_dir: Path | None) -> list[str]:
	if label_dir is None:
		return []
	return sorted(
		item.name
		for item in label_dir.iterdir()
		if item.is_dir() and not item.name.startswith(".") and item.name not in EXCLUDED_FOLDER_NAMES
	)


def readable_class_name(class_name: str) -> tuple[str, str]:
	parts = class_name.split("___", 1)
	if len(parts) == 2:
		plant = parts[0].replace("__", " ").replace("_", " ").strip()
		disease = parts[1].replace("__", " ").replace("_", " ").strip()
		return plant, disease

	parts = class_name.split("__", 1)
	if len(parts) == 2:
		plant = parts[0].replace("_", " ").strip()
		disease = parts[1].replace("__", " ").replace("_", " ").strip()
		return plant, disease

	return "Unknown plant", class_name.replace("__", " ").replace("_", " ").strip()


def disease_solution(disease_name: str) -> str:
	cleaned = disease_name.strip()
	return DISEASE_SOLUTIONS.get(cleaned, "No specific solution was stored for this label. Use proper watering, isolate the plant, and consult an agricultural extension service for confirmation.")


def build_fallback_explanation(predicted_label: str) -> dict[str, str]:
	plant_name, disease_name = readable_class_name(predicted_label)
	return {
		"plant_name": plant_name,
		"disease_name": disease_name,
		"summary": f"The model predicted {predicted_label}.",
		"solution": disease_solution(disease_name),
	}


@st.cache_resource(show_spinner=False)
def get_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
	load_model = importlib.import_module("tensorflow.keras.models").load_model
	return load_model(MODEL_PATH, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
	rgb_image = image.convert("RGB").resize(EXPECTED_SIZE)
	array = np.asarray(rgb_image, dtype=np.float32) / 255.0
	return np.expand_dims(array, axis=0)


def format_probabilities(probabilities: np.ndarray, class_names: list[str]) -> Tuple[str, float, np.ndarray, np.ndarray]:
	flattened = np.asarray(probabilities).squeeze()

	if flattened.ndim == 0:
		positive = float(flattened)
		distribution = np.array([1.0 - positive, positive], dtype=np.float32)
		if len(class_names) == 2:
			labels = np.array(class_names[:2])
		else:
			labels = np.array(["Class 0", "Class 1"])
		index = int(positive >= 0.5)
		label = labels[index]
		confidence = positive if index == 1 else 1.0 - positive
		return label, confidence, labels, distribution

	distribution = flattened.astype(np.float32)
	index = int(np.argmax(distribution))
	labels = np.array(class_names[: len(distribution)]) if class_names else np.array([f"Class {i}" for i in range(len(distribution))])
	label = labels[index] if index < len(labels) else f"Class {index}"
	confidence = float(distribution[index])
	return label, confidence, labels, distribution


def predict_image(model, image: Image.Image, class_names: list[str]):
	batch = preprocess_image(image)
	raw_predictions = model.predict(batch, verbose=0)
	return format_probabilities(raw_predictions[0], class_names)


def build_probability_frame(labels: np.ndarray, distribution: np.ndarray) -> pd.DataFrame:
	frame = pd.DataFrame({"Class": labels, "Probability": distribution})
	return frame.sort_values("Probability", ascending=False).reset_index(drop=True)


def build_result_summary(predicted_label: str) -> tuple[str, str, str]:
	plant_name, disease_name = readable_class_name(predicted_label)
	solution = disease_solution(disease_name)
	return plant_name, disease_name, solution


@st.cache_resource(show_spinner=False)
def get_gemini_model(api_key: str):
	# removed SDK-based client; REST approach used instead
	raise RuntimeError("SDK client not available; use REST path")


def parse_gemini_response(response_text: str) -> dict[str, str]:
	text = response_text.strip()
	if text.startswith("```"):
		text = text.strip("`")
		if text.lower().startswith("json"):
			text = text[4:].strip()
	try:
		payload = json.loads(text)
		return {
			"plant_name": str(payload.get("plant_name", "Unknown plant")),
			"disease_name": str(payload.get("disease_name", "Unknown disease")),
			"summary": str(payload.get("summary", "")),
			"solution": str(payload.get("solution", "")),
		}
	except Exception:
		return {
			"plant_name": "Unknown plant",
			"disease_name": "Unknown disease",
			"summary": text,
			"solution": "",
		}


def generate_gemini_explanation(image: Image.Image, predicted_label: str, confidence: float, api_key: str) -> dict[str, str]:
	# REST path: use the text-bison endpoint with an API key (environment variable recommended)
	prompt = (
		"You are a plant disease assistant. Based on the model prediction and context, "
		"return a JSON object with keys: plant_name, disease_name, summary, and solution. "
		"Keep fields short and practical. If uncertain, set disease_name to 'uncertain'.\n"
		f"Model prediction: {predicted_label}. Confidence: {confidence:.2%}.\n"
		"If the predicted label contains the plant and disease (e.g., 'Tomato_Early_blight'), split them appropriately."
	)
	try:
		return call_gemini_rest(api_key, prompt)
	except Exception:
		return build_fallback_explanation(predicted_label)


def call_gemini_rest(api_key: str, prompt: str) -> dict[str, str]:
	if not api_key:
		raise ValueError("API key required for REST Gemini calls")
	endpoint = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key={api_key}"
	payload = {"prompt": {"text": prompt}, "temperature": 0.2, "maxOutputTokens": 300}
	resp = requests.post(endpoint, json=payload, timeout=30)
	resp.raise_for_status()
	data = resp.json()
	text = ""
	if isinstance(data, dict):
		if "candidates" in data and data["candidates"]:
			text = data["candidates"][0].get("content", "")
		elif "output" in data and data["output"]:
			try:
				text = data["output"][0].get("content", "")
			except Exception:
				text = str(data)
		else:
			text = json.dumps(data)

	m = re.search(r"\{[\s\S]*\}", text)
	json_text = m.group(0) if m else text.strip()
	try:
		parsed = json.loads(json_text)
		return {"plant_name": parsed.get("plant_name", "Unknown plant"), "disease_name": parsed.get("disease_name", "Unknown disease"), "summary": parsed.get("summary", ""), "solution": parsed.get("solution", "")}
	except Exception:
		return {"plant_name": "Unknown plant", "disease_name": "Unknown disease", "summary": text, "solution": ""}


def main():
	st.markdown(
		"""
		<div class="hero">
			<h1>Plant Disease Classifier</h1>
			<p>Upload a leaf image, run the saved model, and inspect the prediction probabilities.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	label_dir = resolve_label_directory()
	class_names = discover_class_names(label_dir)
	generative_api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

	left, right = st.columns([1.05, 1.0], gap="large")

	with left:
		st.markdown('<div class="card">', unsafe_allow_html=True)
		st.subheader("Prediction input")
		uploaded_file = st.file_uploader("Choose a plant leaf image", type=["png", "jpg", "jpeg", "webp"])
		st.markdown("</div>", unsafe_allow_html=True)

		if uploaded_file is None:
			st.info("Upload a leaf image to get started.")
			return

		image = Image.open(uploaded_file)
		st.image(image, use_container_width=True)

	with right:
		st.markdown('<div class="metric-box">', unsafe_allow_html=True)
		st.markdown('<div class="small-label">Model</div>', unsafe_allow_html=True)
		st.write(MODEL_PATH.name)
		st.markdown('<div class="small-label">Expected input</div>', unsafe_allow_html=True)
		st.write(f"{EXPECTED_SIZE[0]} x {EXPECTED_SIZE[1]} x 3")
		st.markdown('<div class="small-label">Detected classes</div>', unsafe_allow_html=True)
		st.write(len(class_names) if class_names else "Unknown")
		if label_dir is not None:
			st.caption(f"Label source: {label_dir.relative_to(ROOT)}")
		if class_names:
			with st.expander("Detected label names", expanded=False):
				for class_name in class_names:
					plant, disease = readable_class_name(class_name)
					sol = find_solution_for_disease(disease)
					st.write(f"- {plant}: {disease}")
					st.caption(sol)
		st.markdown('</div>', unsafe_allow_html=True)

		try:
			model = get_model()
		except Exception as exc:
			st.error(f"Could not load the model: {exc}")
			return

		try:
			predicted_label, confidence, labels, distribution = predict_image(model, image, class_names)
		except Exception as exc:
			st.error(f"Prediction failed: {exc}")
			return

		plant_name, disease_name, solution = build_result_summary(predicted_label)

		st.markdown('<div class="card">', unsafe_allow_html=True)
		st.subheader("Prediction result")
		st.metric("Top prediction", predicted_label, f"{confidence:.2%}")
		st.markdown('<div class="small-label">Plant name</div>', unsafe_allow_html=True)
		st.write(plant_name)
		st.markdown('<div class="small-label">Disease type</div>', unsafe_allow_html=True)
		st.write(disease_name)
		st.markdown('<div class="small-label">Suggested solution</div>', unsafe_allow_html=True)
		st.write(find_solution_for_disease(disease_name))

		if len(distribution) == 2 and not class_names:
			st.info(
				"This saved model is exposing only two output scores, so the app can tell which side it favors, but it cannot attach a disease name until the training label map is provided."
			)
		elif len(distribution) == 2:
			st.info(
				f"The model favors {predicted_label}. If this result looks like the wrong disease name, the saved model may not match the folder labels exactly."
			)

		# If an API key is set in the environment, call Gemini REST automatically and show results.
		if generative_api_key:
			with st.spinner("Generating plant diagnosis with Gemini..."):
				try:
					gemini_result = generate_gemini_explanation(image, predicted_label, confidence, generative_api_key)
				except Exception as exc:
					st.error(f"Gemini request failed: {exc}")
					gemini_result = build_fallback_explanation(predicted_label)

				st.markdown('<div class="metric-box">', unsafe_allow_html=True)
				st.markdown('<div class="small-label">Plant name (Gemini)</div>', unsafe_allow_html=True)
				st.write(gemini_result.get("plant_name", "Unknown plant"))
				st.markdown('<div class="small-label">Disease type (Gemini)</div>', unsafe_allow_html=True)
				st.write(gemini_result.get("disease_name", "Unknown disease"))
				st.markdown('<div class="small-label">What Gemini says</div>', unsafe_allow_html=True)
				st.write(gemini_result.get("summary", ""))
				st.markdown('<div class="small-label">Suggested solution</div>', unsafe_allow_html=True)
				st.write(gemini_result.get("solution", ""))
				st.markdown('</div>', unsafe_allow_html=True)

		if len(distribution) > 1:
			ranking = build_probability_frame(labels, distribution)
			st.dataframe(
				ranking.head(5),
				use_container_width=True,
				hide_index=True,
			)
			st.bar_chart(ranking.set_index("Class")["Probability"])
		else:
			st.progress(float(distribution[0]))
			st.write(f"Confidence: {confidence:.2%}")

		st.markdown("</div>", unsafe_allow_html=True)

	if class_names:
		st.caption("Class order is read from the dataset folders so it stays aligned with training-time label encoding.")
	else:
		st.warning("No class folders were found under dataset/. The app can still load images, but predictions may be unlabeled.")


if __name__ == "__main__":
	main()
