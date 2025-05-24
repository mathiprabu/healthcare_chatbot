import os
import nltk
import ssl
import requests
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": [
            "Hi there",
            "Hello",
            "Hey",
            "I'm fine, thank you",
            "Nothing much",
        ],
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"],
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"],
    },
    {
        "tag": "Fever",
        "patterns": [
            "How do you treat a mild Fever?",
            "fever",
            "Which medicine to take if I get a mild fever?",
        ],
        "responses": [
            "To treat a fever at home: 1) Drink plenty of fluids to stay hydrated, "
            "2) Dress in lightweight clothing, "
            "3) Use a light blanket if you feel chilled, until the chills end, "
            "4) Take acetaminophen (Tylenol, others) or ibuprofen (Advil, Motrin IB, others), "
            "5) Get medical help if the fever lasts more than five days in a row",
        ],
    },
    {
        "tag": "your name",
        "patterns": [
            "what is your name",
            "how can i call you",
            "Tell me, what should I call you",
            "May I know your name, please?",
        ],
        "responses": ["you can call me Rolex", "i'm Rolex"],
    },
    {
        "tag": "your boss",
        "patterns": ["who is your boss", "who created you", "what is your boss name"],
        "responses": ["Akilan, he created me"],
    },
    {
        "tag": "cough",
        "patterns": [
            "How to cure cough?",
            "How do you treat cough?",
            "what to do if i get a cough?",
            "Which medicine to take if I get cough?",
            "How do you get rid of cough?",
        ],
        "responses": [
            "1) Honey: Use honey to treat a cough, mix 2 teaspoons (tsp) with warm water or an herbal tea. "
            "Drink this mixture once or twice a day. Do not give honey to children under 1 year of age. "
            "2) Ginger: Brew up a soothing ginger tea by adding 20–40 grams (g) of fresh ginger slices to a cup of hot water. "
            "Allow to steep for a few minutes before drinking. "
            "Add honey or lemon juice to improve the taste and further soothe a cough. "
            "3) Fluids: Staying hydrated is vital for those with a cough or cold. "
            "Research indicates that drinking liquids at room temperature can alleviate a cough, "
            "runny nose, and sneezing.",
        ],
    },
    {
        "tag": "skin problem",
        "patterns": [
            "How do you treat Skin problems?",
            "what to do if i get a skin allergy?",
            "Which medicine to take if I get a skin allergy?",
            "How to cure skin allergy?",
            "skin problem",
        ],
        "responses": [
            "1) Hydrocortisone cream, "
            "2) Ointments like calamine lotion, "
            "3) Antihistamines, "
            "4) Cold compresses, "
            "5) Oatmeal baths, "
            "6) Talk to your doctor about what's best for your specific rash.",
        ],
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": [
            "Sure, what do you need help with?",
            "I'm here to help. What's the problem?",
            "How can I assist you?",
        ],
    },
    {
        "tag": "choking",
        "patterns": [
            "How do you treat Choking?",
            "what to do if i get a Choke?",
            "Which medicine to take if I get Choked?",
            "How to cure Choking?",
            "choking",
        ],
        "responses": [
            "1) Encourage them to keep coughing to try to clear the blockage. "
            "2) Ask them to try to spit out the object if it's in their mouth. "
            "3) If coughing doesn't work, start back blows and abdominal thrusts.",
        ],
    },
    {
        "tag": "wound",
        "patterns": [
            "How do you treat a wound?",
            "what to do if i get a Wound?",
            "Which medicine to take if I get wounded?",
            "How to cure a wound?",
            "wound",
        ],
        "responses": [
            "1) Rinse the cut or wound with water and apply pressure with sterile gauze. "
            "2) If blood soaks through the bandage, place another bandage on top. "
            "3) Raise the injured body part to slow bleeding. "
            "4) When bleeding stops, cover the wound with a new, clean bandage.",
        ],
    },
    {
        "tag": "bleeding",
        "patterns": [
            "How do you treat bleeding?",
            "what to do if i get a Bleeding?",
            "Which medicine to take if I get bleeding?",
            "How to cure Bleeding?",
            "bleeding",
        ],
        "responses": [
            "1) First aid actions to manage external bleeding include applying direct pressure to the wound, "
            "maintaining the pressure using pads, "
            "bandages, and raising the injured limb above the level of the heart if possible.",
        ],
    },
    {
        "tag": "eye injury",
        "patterns": [
            "How do you treat an eye Injury?",
            "what to do if i get a eye Injury?",
            "Which medicine to take if I injured my eye?",
            "How to cure injured eye?",
        ],
        "responses": [
            "1) DO NOT rub the eye, "
            "2) Blink several times and allow tears to flush out the particle, "
            "3) Lift the upper eyelid over the lashes of your lower lid to let the eyelashes try to brush the particle out, "
            "4) Use eyewash, saline solution, or running tap water to flush the eye out.",
        ],
    },
    {  
        "tag": "teeth",
        "patterns": [
            "How do you treat broken Teeth?",
            "what to do if my Teeth got broken?",
            "Which medicine to take if I get broken teeth?",
            "cure broken teeth?",
        ],   
        "responses": [
            "1) Lick the tooth clean if it's dirty, or rinse it in water. "
            "2) Put it back into position (adult teeth only); "
            "never try to re-insert a baby tooth (see below). "
            "3) Bite on a handkerchief to hold the tooth in place. "
            "4) Go to see a dentist as an emergency.",
        ],
    },
    {
        "tag": "headache",
        "patterns": [
            "How do you treat a mild Headache?",
            "what to do if i get a mild Headache?",
            "Which medicine to take if I have a mild headache?",
            "How to cure a mild headache?",
            "headache",
        ],
        "responses": [
            "Give ibuprofen (Advil, Motrin), aspirin, or acetaminophen (Tylenol) for pain. "
            "Avoid ibuprofen and other NSAIDs if the person has heart failure or kidney failure. "
            "Do not give aspirin to a child under age 18.",
        ],
    },
    {
        "tag": "cold",
        "patterns": [
            "How do you treat a Cold?",
            "what to do if i get a mild Cold?",
            "Which medicine to take if I have a Cold?",
            "How to cure Cold?",
        ],
        "responses": [
            "1) Keeping hydrated is absolutely vital to help 'flush' out the cold, "
            "as well as to break down congestion and keep your throat lubricated. "
            "2) Vitamin C is extremely helpful when fighting infection, "
            "so at the first sign of a cold be sure to increase your intake by eating plenty of berries, "
            "citrus fruits, papayas, broccoli, and red peppers which will help keep you protected. "
            "3) When it comes to combating a cold, Vitamin D is essential in helping to regulate immune response.",
        ],
    },
    {
        "tag": "snake bite",
        "patterns": [
            "How do you treat a snake bite?",
            "what to do if i get a snake bite?",
            "Which medicine to take if I get a snake bite?",
            "How to cure snake bite?",
            "i got bit by a snake",
        ],
        "responses": [
            "While waiting for medical help: 1) Move the person beyond striking distance of the snake. "
            "2) Have the person lie down with wound below the heart. "
            "3) Keep the person calm and at rest, remaining as still as possible to keep venom from spreading. "
            "4) Cover the wound with a loose, sterile bandage. "
            "5) Remove any jewelry from the area that was bitten. "
            "6) Remove shoes if the leg or foot was bitten.",
        ],
    },
    {
        "tag": "corona",
        "patterns": [
            "covid 19",
            "how to cure corona",
        ],
        "responses": [
            "Remdesivir injection is also used to treat coronavirus disease 2019 (COVID-19 infection) "
            "caused by the SARS-CoV-2 virus in non-hospitalized adults and children 28 days and older "
            "who weigh at least 6.6 pounds (3 kg) who are at high risk of progression to severe COVID-19.",
        ],
    },
    {
        "tag": "leg pain",
        "patterns": [
            "leg pain",
            "knee pain",
            "neck pain",
            "hand pain",
        ],
        "responses": [
            "Rest as much as possible. "
            "Elevate your muscle. "
            "Apply ice for up to 15 minutes. Do this 4 times per day, more often for the first few days. "
            "Gently stretch and massage cramping muscles. "
            "Take over-the-counter pain medicines such as acetaminophen (Tylenol) or ibuprofen (Advil, Motrin).",
        ],
    },
    {
        "tag": "stomach pain",
        "patterns": [
            "stomach pain",
            "how to cure stomach pain",
        ],
        "responses": [
            "Take mild painkillers such as paracetamol. "
            "Please check the packet for the right dose. "
            "Avoid aspirin or anti-inflammatory drugs unless advised to take them by a doctor. "
            "These drugs can make some types of abdominal pain worse.",
        ],        
    },
    {
        "tag": "cancer",
        "patterns": [
            "cancer",
            "blood cancer",
            "liver cancer",
            "kidney cancer",
            "lung cancer",
        ],
        "responses": [
            "1.Surgery,2.Chemotherapy,3.Radiation Therapy,4.Targeted Therapy,5.Immunotherapy",
            "6.Stem Cell or Bone Marrow Transplant,7.Hormone Therapy",
        ],
    },
    { 
        "tag": "disease",
        "patterns": [
            "malaria",
            "dengue",
            "chikungunya",
            "yellow fever",
        ],
        "responses": [
            "RTS,S/AS01 (RTS,S) is the first and, to date, "
            "only vaccine that has demonstrated it can significantly reduce disease "
            "in young children living in moderate-to-high disease transmission areas. "
            "It acts against the Plasmodium falciparum parasite.",
        ],
    },
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
print(x.shape)
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response

def main():
    st.title("HEALTHCARE CHATBOT")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

if __name__ == "__main__":
    main()