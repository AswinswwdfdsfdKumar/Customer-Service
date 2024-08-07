import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from fuzzywuzzy import process
import streamlit as st

# Load the Excel file and prepare the data
file_path = "RPA .xlsx"
df_responses = pd.read_excel(file_path, sheet_name='Predifined Script',
                             engine='openpyxl')

# Prepare predefined responses dictionary
predefined_responses = {}
for index, row in df_responses.iterrows():
    predefined_responses[row['Common_Inquiry']] = row['Response']

predefined_responses_dict = {}
for inquiry, response in predefined_responses.items():
    category = 'Uncategorized'
    predefined_responses_dict.setdefault(category, {})[inquiry] = response

# Load and prepare the training data
df_training = pd.read_excel(file_path, sheet_name='Predifined Script',
                            engine='openpyxl')
df_training.dropna(subset=['Common_Inquiry', 'Response'], inplace=True)

# Train the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df_training['Common_Inquiry'], df_training['Response'])
joblib.dump(model, 'email_classifier.pkl')

# Load the trained model
model = joblib.load('email_classifier.pkl')


# Function to check predefined responses
def check_predefined_responses(text):
    for category, responses in predefined_responses_dict.items():
        best_match, score = process.extractOne(text, responses.keys())
        if score >= 80:
            return responses[best_match]
    return None


# Function to classify email
def classify_email(text):
    response = check_predefined_responses(text)
    if response:
        return response

    category = model.predict([text])[0]
    return next(iter(predefined_responses_dict.get(category, {}).values()),
                "Sorry, I don't have an answer for that.")


# Main function for Streamlit app
def main():
    st.title('Email Inquiry Classifier')

    # Text input for the inquiry
    inquiry = st.text_input("Enter your inquiry:")

    if st.button("Classify"):
        if inquiry:
            response = classify_email(inquiry)
            st.write("Response:", response)
        else:
            st.write("Please enter an inquiry.")


if __name__ == "__main__":
    main()
