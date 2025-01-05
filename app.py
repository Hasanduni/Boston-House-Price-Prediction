import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Set up the page title
st.title("Boston House Price Prediction")

# App Introduction
st.write("""
### Welcome to the Boston House Price Prediction App!
This app allows users to predict the median value of owner-occupied homes in Boston based on key features 
of the area such as crime rate, nitric oxide levels, and average number of rooms. Simply input the values for 
the features, and the model will provide an estimated house price.
""")

# Custom CSS to change the page background color
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f8ff;
        }
        .sidebar .sidebar-content {
            background-color: #f0f8ff;
        }
        .title {
            color: #4b0082;
        }
        .stButton>button {
            background-color: #4b0082;
            color: white;
        }
        .stButton>button:hover {
            background-color: #551a8b;
        }
    </style>
""", unsafe_allow_html=True)

# Boston Dataset Description
with st.expander("About the Dataset"):
    st.write("""
    The **Boston House Prices Dataset** is a classic dataset in machine learning often used for regression problems. 
    It includes data collected by the U.S. Census Service concerning housing in the Boston area.
    """)
    st.write("""
    **Dataset Characteristics:**
    - **Number of Instances:** 506
    - **Number of Attributes:** 13 predictive attributes + 1 target (Median Value of homes)
    - **Missing Values:** None
    """)
    st.write("""
    **Attributes:**
    1. **CRIM:** Per capita crime rate by town  
    2. **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.  
    3. **INDUS:** Proportion of non-retail business acres per town  
    4. **CHAS:** Charles River dummy variable (1 if bounds river, 0 otherwise)  
    5. **NOX:** Nitric oxides concentration (parts per 10 million)  
    6. **RM:** Average number of rooms per dwelling  
    7. **AGE:** Proportion of owner-occupied units built prior to 1940  
    8. **DIS:** Weighted distances to five Boston employment centres  
    9. **RAD:** Index of accessibility to radial highways  
    10. **TAX:** Full-value property-tax rate per $10,000  
    11. **PTRATIO:** Pupil-teacher ratio by town  
    12. **B:** 1000(Bk - 0.63)^2, where Bk is the proportion of black people by town  
    13. **LSTAT:** Percentage of lower-status population  
    14. **MEDV:** Median value of owner-occupied homes in $1000's (Target Variable)  
    """)
    st.caption("""
    **Source:** Harrison, D. and Rubinfeld, D.L., "Hedonic prices and the demand for clean air", J. Environ. Economics & Management, 1978.  
    [Link to Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)
    """)

# Feature Descriptions
with st.expander("Feature Descriptions"):
    st.write("Below are the descriptions of each feature used in this app:")
    columns = [
        ("CRIM", "Per capita crime rate by town"),
        ("ZN", "Proportion of residential land zoned for lots over 25,000 sq.ft."),
        ("INDUS", "Proportion of non-retail business acres per town"),
        ("CHAS", "Charles River dummy variable (1 if bounds river, 0 otherwise)"),
        ("NOX", "Nitric oxides concentration (parts per 10 million)"),
        ("RM", "Average number of rooms per dwelling"),
        ("AGE", "Proportion of owner-occupied units built prior to 1940"),
        ("DIS", "Weighted distances to five Boston employment centres"),
        ("RAD", "Index of accessibility to radial highways"),
        ("TAX", "Full-value property-tax rate per $10,000"),
        ("PTRATIO", "Pupil-teacher ratio by town"),
        ("B", "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town"),
        ("LSTAT", "% lower status of the population"),
    ]
    for feature, desc in columns:
        st.markdown(f"**{feature}:** {desc}")

# Create placeholders for inputs
st.write("### Enter the Features:")
input_values = []
for i in range(0, len(columns), 4):
    col1, col2, col3, col4 = st.columns(4)
    for col, (feature, desc) in zip([col1, col2, col3, col4], columns[i:i+4]):
        with col:
            st.markdown(f"**{feature}**")
            value = st.text_input(f"Enter value for {feature}:", placeholder=desc)
            input_values.append(value)

# Filter out None or empty values and convert to float
input_values = [float(x) for x in input_values if x]

# Predict when the button is clicked
if st.button("Predict"):
    if len(input_values) == 13:  # Ensure all fields are filled
        # Scale the input
        scaled_data = scalar.transform(np.array(input_values).reshape(1, -1))

        # Predict
        prediction = regmodel.predict(scaled_data)[0]

        # Display the prediction
        st.success(f"The predicted house price is: ${prediction:.2f}K USD")
    else:
        st.error("Please fill all the required fields!")
