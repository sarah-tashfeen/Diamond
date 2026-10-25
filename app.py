import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")
st.title("💎 Diamond Price Predictor")
st.write("Enter the diamond's specifications to predict its price (US dollars).")
st.caption("Model: Hist Gradient Boosting, selected by 10-fold cross validation on 53,775 diamonds (typical error about $548).")

@st.cache_resource
def load_model():
    with open("diamond_price_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat", 0.2, 5.1, 0.70, 0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color (D = best)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=3)
with col2:
    depth = st.number_input("Depth %", 40.0, 80.0, 61.5, 0.1)
    table = st.number_input("Table %", 40.0, 100.0, 57.0, 0.1)
    x = st.number_input("Length x (mm)", 1.0, 11.0, 5.70, 0.01)
    y = st.number_input("Width y (mm)", 1.0, 60.0, 5.73, 0.01)
    z = st.number_input("Depth z (mm)", 1.0, 32.0, 3.52, 0.01)

if st.button("Predict Price"):
    row = pd.DataFrame([{"carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
                         "cut": cut, "color": color, "clarity": clarity}])
    price = model.predict(row)[0]
    st.success(f"Predicted price: ${price:,.2f}")
