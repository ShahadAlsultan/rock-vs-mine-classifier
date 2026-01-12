import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Data
# -----------------------------
sonar_data = pd.read_csv(r"C:\Users\alsul\Desktop\sea\Copy of sonar data.csv", header=None)

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM model
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train_scaled, Y_train)

# -----------------------------
# Streamlit UI Styling
# -----------------------------
st.set_page_config(page_title="Rock vs Mine Classifier", page_icon="🔍", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #4A90E2;
            font-size: 40px;
            font-weight: bold;
        }
        .sub-text {
            text-align: center;
            color: #555;
            font-size: 18px;
        }
        .rock-box {
            background-color: #D0E8FF;
            padding: 15px;
            border-radius: 10px;
            color: #005BBB;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        }
        .mine-box {
            background-color: #FFE0E0;
            padding: 15px;
            border-radius: 10px;
            color: #BB0000;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Image
# -----------------------------
st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=120)

# Title
st.markdown("<div class='main-title'>Rock vs Mine Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Paste SONAR values below and let the model classify the object</div>", unsafe_allow_html=True)

st.write("---")

# -----------------------------
# Input Box (Paste 60 numbers)
# -----------------------------
st.subheader("Paste SONAR Features")

user_input = st.text_area(
    "Paste 60 numbers separated by spaces or commas:",
    placeholder="0.03 0.12 0.55 0.88 ..."
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict", type="primary"):
    try:
        # Convert text to list of floats
        values = [float(x) for x in user_input.replace(",", " ").split()]
        
        if len(values) != 60:
            st.error(f"❌ You entered {len(values)} values. You must enter exactly 60.")
        else:
            input_array = np.asarray(values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)

            st.write("---")
            st.subheader("Prediction Result")

            if prediction[0] == 'R':
                st.markdown("<div class='rock-box'>🪨 The object is classified as: ROCK</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='mine-box'>💣 The object is classified as: MINE</div>", unsafe_allow_html=True)

    except:
        st.error("❌ Invalid input. Make sure all values are numbers.")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.write("---")
st.subheader("Model Evaluation")

if st.button("Evaluate Model"):
    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    acc = accuracy_score(Y_test, y_pred)
    st.write(f"### ✅ Accuracy: **{acc:.4f}**")

    # Confusion Matrix
    st.write("### 📊 Confusion Matrix")
    cm = confusion_matrix(Y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Mine", "Rock"], yticklabels=["Mine", "Rock"])
    st.pyplot(fig)

    # Classification Report
    st.write("### 📄 Classification Report")
    report = classification_report(Y_test, y_pred, output_dict=True)
    st.dataframe(report)
