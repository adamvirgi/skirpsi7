import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.joblib')

# Define the application
def main():
    # Create a title for the application
    st.title('Stunting Status Prediction App')

    # Create a form to collect user input
    with st.form('user_input'):
        age = st.number_input('Age (Month)', min_value=0, max_value=72)
        gender = st.radio('Gender', ['Female', 'Male'])
        body_height = st.number_input('Body Height (cm)', min_value=0.0, max_value=100.0)
        body_weight = st.number_input('Body Weight (kg)', min_value=0.0, max_value=50.0)

        # Submit button
        submit = st.form_submit_button('Predict')

    # Make predictions if the submit button is clicked
    if submit:
        # Preprocess the user input
        data = {'Age (Month)': [age],
                'Gender': [gender],
                'Body height': [body_height],
                'Body weight': [body_weight]}
        df = pd.DataFrame(data)

        # Make predictions
        prediction = model.predict(df)[0]

        # Display the prediction
        st.write(f'Predicted Status: {prediction}')

# Run the application
if __name__ == '__main__':
    main()
