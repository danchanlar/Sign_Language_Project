#writefile streamlit_app.py
import json
import tempfile
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="GSL Public Service Assistant",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_CONFIG_PATH = Path("/content/drive/MyDrive/GSL_public_service_outputs/models/app_config.json")

mp_holistic = mp.solutions.holistic


# ============================================================
# Feature extraction
# ============================================================

def landmarks_to_array(landmarks, n_landmarks, dims):
    if landmarks is None:
        return np.zeros(n_landmarks * dims, dtype=np.float32)

    arr = []
    for lm in landmarks.landmark:
        values = [lm.x, lm.y, lm.z]
        if dims == 4:
            values.append(lm.visibility)
        arr.extend(values)

    return np.array(arr, dtype=np.float32)


def extract_keypoints_from_results(results, feature_mode="hands_pose"):
    left_hand = landmarks_to_array(results.left_hand_landmarks, 21, 3)
    right_hand = landmarks_to_array(results.right_hand_landmarks, 21, 3)

    if feature_mode == "hands_only":
        return np.concatenate([left_hand, right_hand])

    if feature_mode == "hands_pose":
        pose = landmarks_to_array(results.pose_landmarks, 33, 4)
        return np.concatenate([left_hand, right_hand, pose])

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def sample_frame_indices(total_frames, sequence_length):
    if total_frames <= 0:
        return list(range(sequence_length))

    if total_frames >= sequence_length:
        return np.linspace(0, total_frames - 1, sequence_length).astype(int).tolist()

    indices = list(range(total_frames))
    while len(indices) < sequence_length:
        indices.append(total_frames - 1)

    return indices[:sequence_length]


def extract_video_features(video_path, sequence_length=60, feature_mode="hands_pose", resize_width=640):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, sequence_length)

    sequence = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok:
                dim = 126 if feature_mode == "hands_only" else 258
                sequence.append(np.zeros(dim, dtype=np.float32))
                continue

            if resize_width is not None:
                h, w = frame.shape[:2]
                scale = resize_width / w
                frame = cv2.resize(frame, (resize_width, int(h * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            keypoints = extract_keypoints_from_results(results, feature_mode)
            sequence.append(keypoints)

    cap.release()
    return np.array(sequence, dtype=np.float32)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_resources(config_path):
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = tf.keras.models.load_model(config["model_path"])
    encoder = joblib.load(config["encoder_path"])

    norm = np.load(config["normalization_path"])
    mean = norm["mean"]
    std = norm["std"]

    natural_map_path = Path(config.get("natural_greek_map_path", ""))
    if natural_map_path.exists():
        with open(natural_map_path, "r", encoding="utf-8") as f:
            natural_map = json.load(f)
    else:
        natural_map = {}

    return config, model, encoder, mean, std, natural_map


def predict_video(video_path, config, model, encoder, mean, std, natural_map, top_k=5):
    sequence_length = int(config["sequence_length"])
    feature_mode = config["feature_mode"]

    features = extract_video_features(
        video_path,
        sequence_length=sequence_length,
        feature_mode=feature_mode
    )

    X = features.reshape(1, sequence_length, -1)
    X = (X - mean) / std

    probs = model.predict(X, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    rows = []
    for idx in top_indices:
        gloss = encoder.inverse_transform([idx])[0]
        rows.append({
            "GSL gloss sentence": gloss,
            "Natural Greek meaning": natural_map.get(gloss, gloss),
            "Confidence": float(probs[idx])
        })

    return pd.DataFrame(rows)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Settings")

config_path = st.sidebar.text_input(
    "Model config path",
    value=str(DEFAULT_CONFIG_PATH)
)

top_k = st.sidebar.slider(
    "Top predictions",
    min_value=1,
    max_value=10,
    value=5
)

confidence_threshold = st.sidebar.slider(
    "Minimum confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05
)

service_context = st.sidebar.selectbox(
    "Public-service scenario",
    [
        "Citizen Service Center",
        "Police / Public Safety",
        "Municipality",
        "Hospital / Health Service",
        "School Office",
        "Other Public Service"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app works only for the predefined GSL sentence classes included in the trained model."
)


# ============================================================
# Main page
# ============================================================

st.title("🤟 Greek Sign Language Public-Service Assistant")

st.markdown(
    """
This application recognizes **predefined Greek Sign Language sentence inputs** related to public-service communication.

It is designed as a **closed-domain assistant** for scenarios such as citizen service centers, police services,
municipal offices, hospitals and school offices.

It is **not** a general sign language translator and does not replace a professional interpreter.
"""
)

try:
    config, model, encoder, mean, std, natural_map = load_resources(config_path)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model/config: {e}")
    st.stop()


with st.expander("ℹ️ How to use this app", expanded=True):
    st.markdown(
        """
1. Upload a short video containing one GSL public-service sentence.
2. Press **Recognize sentence**.
3. The app will show:
   - the predicted GSL gloss sentence,
   - the possible natural Greek meaning,
   - the confidence score,
   - the top alternative predictions,
   - a suggested response for the public-service employee.

For best results, the signer should be clearly visible, with good lighting and visible hands.
"""
    )


col1, col2 = st.columns([1.05, 1])

with col1:
    st.subheader("📤 Upload GSL input")

    uploaded_file = st.file_uploader(
        "Upload a short Greek Sign Language video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_video_path = tmp.name

        st.video(temp_video_path)

        recognize = st.button("🔍 Recognize sentence", use_container_width=True)

        if recognize:
            with st.spinner("Processing video with MediaPipe and running prediction..."):
                try:
                    results = predict_video(
                        temp_video_path,
                        config,
                        model,
                        encoder,
                        mean,
                        std,
                        natural_map,
                        top_k=top_k
                    )

                    st.session_state["results"] = results

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


with col2:
    st.subheader("📌 Recognition result")

    if "results" not in st.session_state:
        st.info("Upload a video and press **Recognize sentence**.")
    else:
        results = st.session_state["results"]
        best = results.iloc[0]
        confidence = float(best["Confidence"])

        if confidence >= confidence_threshold:
            st.success("High-confidence prediction")
        else:
            st.warning(
                "Low-confidence prediction. The result may be unreliable. "
                "Try uploading a clearer video."
            )

        st.markdown("### Predicted GSL gloss sentence")
        st.code(best["GSL gloss sentence"], language="text")

        st.markdown("### Possible natural Greek meaning")
        st.info(best["Natural Greek meaning"])

        st.markdown("### Confidence")
        st.progress(min(confidence, 1.0))
        st.write(f"{confidence:.2%}")

        st.markdown("### Top predictions")

        display_df = results.copy()
        display_df["Confidence %"] = display_df["Confidence"].apply(lambda x: round(x * 100, 2))
        st.dataframe(
            display_df[["GSL gloss sentence", "Natural Greek meaning", "Confidence %"]],
            use_container_width=True
        )

        chart_df = display_df.set_index("GSL gloss sentence")[["Confidence %"]]
        st.bar_chart(chart_df)

        st.markdown("### Suggested public-service employee response")

        responses = {
            "Citizen Service Center": "Κατάλαβα. Θα σας βοηθήσω με το αίτημά σας.",
            "Police / Public Safety": "Κατάλαβα. Θα ενημερώσω το αρμόδιο προσωπικό για να σας εξυπηρετήσει.",
            "Municipality": "Κατάλαβα. Θα σας καθοδηγήσω στην αρμόδια υπηρεσία του Δήμου.",
            "Hospital / Health Service": "Κατάλαβα. Θα ενημερώσω άμεσα το αρμόδιο υγειονομικό προσωπικό.",
            "School Office": "Κατάλαβα. Θα σας βοηθήσω με το ζήτημα της σχολικής υπηρεσίας.",
            "Other Public Service": "Κατάλαβα. Παρακαλώ περιμένετε λίγο για να σας εξυπηρετήσουμε."
        }

        suggested_response = responses.get(service_context, responses["Other Public Service"])
        st.success(suggested_response)

        result_json = {
            "predicted_gloss_sentence": best["GSL gloss sentence"],
            "natural_greek_meaning": best["Natural Greek meaning"],
            "confidence": confidence,
            "service_context": service_context,
            "suggested_employee_response": suggested_response
        }

        st.download_button(
            label="⬇️ Download result as JSON",
            data=json.dumps(result_json, ensure_ascii=False, indent=2),
            file_name="gsl_recognition_result.json",
            mime="application/json",
            use_container_width=True
        )


st.markdown("---")
st.caption(
    "Closed-domain GSL public-service assistant. "
    "This application recognizes only the sentence classes included in the trained dataset subset."
)