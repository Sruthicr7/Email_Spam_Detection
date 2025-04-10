import pickle
import streamlit as st

with open(
    r'C:\Users\Hp\OneDrive\Desktop\EI System\logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'C:\Users\Hp\OneDrive\Desktop\EI System\tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    feature_extraction = pickle.load(vectorizer_file)

def predict_mail(input_mail):
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        return 'Ham mail'
    else:
        return 'Spam mail'

def main():
    st.title("Email Spam Detection")
    st.write("Enter an email message to check if it's spam or ham:")

    user_input = st.text_area("Email Message")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.write("Please enter a message.")
        else:
            prediction = predict_mail([user_input])
            st.write(f'The entered message is classified as: {prediction}')

if __name__ == '__main__':
    main()
