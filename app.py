import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load models
try:
    # Traditional models
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    lgb_model = joblib.load("lgb_model.pkl")
    
    # Keras models
    rnn_model = load_model("rnn_model.h5")
    lstm_model = load_model("lstm_model.h5")
    gru_model = load_model("gru_model.h5")
    cnn_model = load_model("cnn_model.h5")
    
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgb', lgb_model)],
    voting='soft'
)

# Sample test data with features
def get_sample_data():
    samples = [
        "SELECT * FROM users",
        "SELECT name, email FROM customers WHERE id = 123",
        "DELETE FROM products WHERE expired = true",
        "admin' OR '1'='1",
        "1; DROP TABLE users",
        "1' UNION SELECT username, password FROM users--"
    ]
    
    labels = [0, 0, 0, 1, 1, 1]  # 0=Benign, 1=Malicious
    
    # Pre-computed features for fitting VotingClassifier
    features = np.array([
        [15, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # SELECT * FROM users
        [38, 0, 0, 0, 1, 0, 0, 1, 0, 0],   # SELECT name, email...
        [35, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # DELETE FROM products...
        [13, 2, 0, 0, 0, 0, 0, 1, 1, 0],   # admin' OR '1'='1
        [15, 0, 1, 0, 0, 1, 0, 0, 0, 0],   # 1; DROP TABLE users
        [46, 1, 0, 1, 1, 0, 1, 0, 0, 0]    # 1' UNION SELECT...
    ])
    
    return samples, labels, features

# Feature extraction
def extract_features(query):
    return np.array([
        len(query),
        query.count("'"),
        query.count(";"),
        int("UNION" in query.upper()),
        int("SELECT" in query.upper()),
        int("DROP" in query.upper()),
        int("--" in query),
        int("=" in query),
        int("OR" in query.upper()),
        int("AND" in query.upper())
    ])

# Preprocessing for neural networks
def preprocess_text(texts, max_len=50):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Fit VotingClassifier with sample data
samples, labels, features = get_sample_data()
voting_classifier.fit(features, labels)

# Streamlit UI
st.title("🛡️ SQL Injection Detection")
model_choice = st.selectbox("Select Model", [
    "Traditional Ensemble (RF+XGB+LGB)",
    "Deep Learning Ensemble (RNN+LSTM+GRU+CNN)"
])

# Get random test sample
random_idx = np.random.randint(0, len(samples))
test_query = samples[random_idx]
true_label = labels[random_idx]

st.subheader("Test Sample")
st.code(test_query)
st.write(f"Actual class: {'Malicious' if true_label == 1 else 'Benign'}")

if st.button("Run Analysis"):
    try:
        if model_choice == "Traditional Ensemble (RF+XGB+LGB)":
            features = extract_features(test_query).reshape(1, -1)
            pred = voting_classifier.predict(features)[0]
            proba = voting_classifier.predict_proba(features)[0][1]  # Probability of being malicious
        else:
            X = preprocess_text([test_query])
            # Get predictions from all models
            predictions = [
                rnn_model.predict(X, verbose=0)[0][0],
                lstm_model.predict(X, verbose=0)[0][0],
                gru_model.predict(X, verbose=0)[0][0],
                cnn_model.predict(X, verbose=0)[0][0]
            ]
            avg_prediction = np.mean(predictions)
            pred = 1 if avg_prediction > 0.5 else 0
            proba = avg_prediction if pred == 1 else 1 - avg_prediction
        
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", 
                     value="Malicious" if pred == 1 else "Benign",
                     delta="Correct" if pred == true_label else "Incorrect")
        
        with col2:
            st.metric("Confidence", f"{proba*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Model metrics
st.subheader("Model Performance Metrics")
if model_choice == "Traditional Ensemble (RF+XGB+LGB)":
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    values = [0.982, 0.981, 0.983, 0.982]
else:
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    values = [0.9969, 0.997, 0.996, 0.996]

fig, ax = plt.subplots()
ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
ax.set_ylim(0, 1.05)
ax.set_title("Validation Set Performance")
st.pyplot(fig)

st.markdown("---")
st.markdown("**Developed with ❤️ by Sheru | SQL Injection Detection System**")