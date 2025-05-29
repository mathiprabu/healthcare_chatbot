import os
import nltk
import ssl
import requests
import streamlit as st
import random
import re
import nltk

# --- Session State Initialization ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
    
if 'topics' not in st.session_state:
    st.session_state.topics = {
        "Health Issues:": ["Bleeding", "Cancer","Choking","Cold","Corona","Cough","Eye injury","Fever","Headache","Leg pain","Skin problem","Snake bite","Stomach pain","Teeth Issue","Wound"],
        "Test Level Tags:": ["BMI check","Cholesterol","Bp level","Hemoglobin","Ptinr level","Salt level","Sugar level"]
    }

if 'show_topics' not in st.session_state:
    st.session_state.show_topics = False

# --- Buttons ---
col1, col2 = st.columns(2)

with col1:
    if st.button("📋 Show Topics"):
        st.session_state.show_topics = not st.session_state.show_topics

with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.conversation = []
        st.rerun()  # Refresh the app

# --- Show Categories and Topics only when the button is clicked ---
if st.session_state.show_topics:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Health Issues:")
        for topic in st.session_state.topics.get("Health Issues:", []):
            st.markdown(f"- {topic}")

    with col2:
        st.markdown("### Test Level Tags:")
        for topic in st.session_state.topics.get("Test Level Tags:", []):
            st.markdown(f"- {topic}")

# --- Display Conversation in Chat-Like Format ---
for chat in st.session_state.conversation:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["ai"])



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
        "name",
            "what is your name",
            "how can i call you",
            "Tell me, what should I call you",
            "May I know your name, please?",
        ],
        "responses": ["you can call me Rolex", "i'm Rolex"],
    },
    {
        "tag": "your boss",
        "patterns": ["boss","who is your boss", "who created you", "what is your boss name"],
        "responses": ["Vashnavi R, She created me"],
    },
    {
        "tag": "cough",
        "patterns": [
        "cough",
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
        "skin",
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
        "choking",
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
        "wound",
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
        "bleeding",
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
        "eye",
        "eye injury",
        
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
        "teeth","teeth issue",
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
        "headache",
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
        "cold",
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
        "tag": "corona",
        "patterns": [
        "corona",
            "covid 19",
            "covid"
            "how to cure corona",
        ],
        "responses": [
            "Remdesivir injection is also used to treat coronavirus disease 2019 (COVID-19 infection) "
            "caused by the SARS-CoV-2 virus in non-hospitalized adults and children 28 days and older "
            "who weigh at least 6.6 pounds (3 kg) who are at high risk of progression to severe COVID-19.",

        ],
    },
    {
        "tag": "snake bite",
        "patterns": [
        "snake bite",
        "snake",
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
        "tag": "leg pain",
        "patterns": [
        "pain",
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
        "tag": "mosquito-borne diseases(malaria,dengue,chikungunya,yellow fever",
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
{
    "tag": "BMI Check",
    "patterns": [
    "bmi",
        "bmi check",
        "calculate bmi",
        "find my bmi",
        "bmi calculation",
    ],
    "responses": [
        "Sure! To calculate BMI, type your weight (kg) and height (cm) like this: bmi 70 170.",
    ],
},

{
    "tag": "sugar level",
    "patterns": [
    "sugar",
        "sugar level",
        "blood sugar",
        "glucose level",
        "what is normal sugar level",
        "high sugar",
        "low sugar",
    ],
    "responses": [
        "Normal fasting blood sugar: 70-99 mg/dL.\n"
        "Post-meal (2 hours) blood sugar: less than 140 mg/dL.\n"
        "Prediabetes: 100-125 mg/dL (fasting).\n"
        "Diabetes: 126 mg/dL or higher (fasting).",
    ],
},

{
    "tag": "bp level",
    "patterns": [
    "BP",
        "bp level",
        "blood pressure",
        "normal blood pressure",
        "high blood pressure",
        "low blood pressure",
    ],
    "responses": [
        "Normal BP: 120/80 mmHg.\n"
        "Elevated: 120-129/<80 mmHg.\n"
        "Hypertension stage 1: 130-139/80-89 mmHg.\n"
        "Hypertension stage 2: ≥140/90 mmHg.\n"
        "Low BP: less than 90/60 mmHg.",
    ],
},
{
    "tag": "ptinr level",
    "patterns": [
    "ptinr level",
        "ptinr",
        "pt inr",
        "prothrombin time",
        "normal ptinr level",
        "blood clotting",
    ],
    "responses": [
        "Normal PT-INR: 0.8 to 1.1.\n"
        "For patients on warfarin (blood thinner), a typical target INR is 2.0 to 3.0.\n"
        "Higher INR means blood clots slower; lower INR means it clots faster.",
    ],
},
{
    "tag": "salt level",
    "patterns": [
    "salt",
        "salt level",
        "sodium level",
        "normal sodium level",
        "hyponatremia",
        "hypernatremia",
    ],
    "responses": [
        "Normal sodium (salt) level: 135 to 145 mEq/L.\n"
        "Hyponatremia (low sodium): less than 135 mEq/L.\n"
        "Hypernatremia (high sodium): greater than 145 mEq/L.",
    ],
},

{
        "tag": "cholesterol",
        "patterns": [
        "cholesterol","cholesterol level",
            "What is normal cholesterol level?",
            "Cholesterol levels",
            "HDL, LDL, Total cholesterol levels",
            "cholesterol",
        ],
        "responses": [
            "Normal cholesterol levels:\n"
            "Total cholesterol: less than 200 mg/dL\n"
            "LDL (bad cholesterol): less than 100 mg/dL\n"
            "HDL (good cholesterol): 40 mg/dL or higher (men), 50 mg/dL or higher (women)\n"
            "Triglycerides: less than 150 mg/dL",
        ],
    },
    {
        "tag": "hemoglobin",
        "patterns": [
        "hemoglobin",
        "hemoglobin level",
            "What is normal hemoglobin level?",
            "Hemoglobin levels",
            "Hb level",
            "hemoglobin",
        ],
        "responses": [
            "Normal hemoglobin (Hb) levels:\n"
            "Men: 13.8 to 17.2 g/dL\n"
            "Women: 12.1 to 15.1 g/dL\n"
            "Children: 11 to 16 g/dL",
        ],
    },
    {
        "tag": "vitamin",
        "patterns": [
            "What is normal vitamin levels?",
            "Vitamin D and B12 levels",
            "vitamin D",
            "vitamin B12",
            "vitamin",
            "vitamin level",
        ],
        "responses": [
            "Normal vitamin levels:\n"
            "Vitamin D: 30 to 100 ng/mL (optimal range)\n"
            "Vitamin B12: 190 to 950 pg/mL",
        ],
    },

]


def chatbot(user_input):
    user_input = user_input.lower()
    
    # Greeting detection
    greetings = ["hi", "hello", "hey", "how are you", "what's up"]
    if any(greet in user_input for greet in greetings):
        # Get all unique tags except small talk intents
        exclude_tags = ["greeting", "thanks", "goodbye", "your name", "your boss", "help"]
        disease_tags = [intent["tag"].capitalize() for intent in intents if intent["tag"] not in exclude_tags]
        return "Hi there! Please click the Show Topics button to view the topics we support.\n\nWhat would you like to know about?"
        
    # BMI calculation logic
    if user_input.startswith("bmi"):
        try:
            parts = user_input.split()
            weight = float(parts[1])
            height_cm = float(parts[2])
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)
            category = ""
            if bmi < 18.5:
                category = "Underweight"
            elif 18.5 <= bmi < 24.9:
                category = "Normal weight"
            elif 25 <= bmi < 29.9:
                category = "Overweight"
            else:
                category = "Obesity"
            return f"Your BMI is {bmi:.2f} ({category})."
        except:
            return "Please provide your weight (kg) and height (cm) in this format: bmi 70 170"
    
    # Pattern matching for intents
    for intent in intents:
        for pattern in intent["patterns"]:
           if re.search(rf'\b{re.escape(pattern.lower())}\b', user_input.lower()):
                return random.choice(intent["responses"])
    
    # Default response
    return "I'm sorry, I don't understand. Can you please rephrase?"

def main():
    st.title("💬 HEALTHCARE CHATBOT 💬")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=300, max_chars=None)

if __name__ == "__main__":
    main()
