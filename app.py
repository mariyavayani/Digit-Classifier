import streamlit as st
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Classifier", page_icon="🔢")
st.title("🔢 Handwritten Digit Classifier")
st.write("A Logistic Regression model trained to recognize handwritten digits (0-9).")

@st.cache_resource
def train_model():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model, X_test, y_test, digits

model, X_test, y_test, digits = train_model()

accuracy = model.score(X_test, y_test)
st.metric("Test Accuracy", f"{accuracy:.2%}")

st.subheader("Try a random test image")
if st.button("Pick a random digit"):
    import random
    idx = random.randint(0, len(X_test) - 1)
    image = X_test[idx].reshape(8, 8)
    prediction = model.predict([X_test[idx]])[0]
    actual = y_test[idx]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Predicted: {prediction} | Actual: {actual}")
    ax.axis("off")
    st.pyplot(fig)

st.subheader("Confusion Matrix")
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
fig2, ax2 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax2)
st.pyplot(fig2)