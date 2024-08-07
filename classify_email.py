import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from fuzzywuzzy import process
import argparse

file_path = r"C:\Users\Aswin Kumar\Documents\RPA .xlsx"
df_responses = pd.read_excel(file_path, sheet_name='Predifined Script', 
                             engine='openpyxl')
predefined_responses = {}
for index, row in df_responses.iterrows():
    predefined_responses[row['Common_Inquiry']] = row['Response']

predefined_responses_dict = {}
for inquiry, response in predefined_responses.items():
    category = 'Uncategorized'
    predefined_responses_dict.setdefault(category, {})[inquiry] = response

df_training = pd.read_excel(file_path, sheet_name='Predifined Script', 
                            engine='openpyxl')

# Remove rows with NaN values
df_training.dropna(subset=['Common_Inquiry', 'Response'], inplace=True)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df_training['Common_Inquiry'], df_training['Response'])
joblib.dump(model, 'email_classifier.pkl')

model = joblib.load('email_classifier.pkl')


def check_predefined_responses(text):
    for category, responses in predefined_responses_dict.items():
        best_match, score = process.extractOne(text, responses.keys())
        if score >= 80:
            return responses[best_match]
    return None


def classify_email(text):
    response = check_predefined_responses(text)
    if response:
        return response

    category = model.predict([text])[0]
    return next(iter(predefined_responses_dict.get(category, {}).values()),
                "Sorry, I don't have an answer for that.")


# Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Classify an email inquiry.")
    parser.add_argument('inquiry', type=str, help='The email inquiry')
    args = parser.parse_args()
    response = classify_email(args.inquiry)
    print(response)


if __name__ == "__main__":
    main()
