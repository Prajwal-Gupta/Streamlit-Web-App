import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier4.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(age,sex,cp,trtbps,restecg,thalachh,exng,oldpeak,slp,caa,thall):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: cp
        in: query
        type: number
        required: true
      - name: trtbps
        in: query
        type: number
        required: true
      - name: restecg
        in: query
        type: number
        required: true
      - name: thalachh
        in: query
        type: number
        required: true
      - name: exng
        in: query
        type: number
        required: true
      - name: oldpeak
        in: query
        type: number
        required: true
      - name: slp
        in: query
        type: number
        required: true
      - name: caa
        in: query
        type: number
        required: true
      - name: thall
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[age,sex,cp,trtbps,restecg,thalachh,exng,oldpeak,slp,caa,thall]])
    print(prediction)
    return prediction



def main():
    st.title("Heart Attack Predictor")
    html_temp = """
    <div style="background-color:PaleVioletRed;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Heart Attack Analysis ML App </h2>
    </div>
    """
    html_temp2 = """
    <div style="background-color:LightCoral;padding:10px">
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # st.header('header')
    st.text('')
    st.text('''
    Enter the details and this will predict if
    there are chances of heart attack with upto 82% accuracy
    ''')
    # st.text('')
    st.markdown(html_temp2,unsafe_allow_html=True)
    st.text('')
    st.subheader("Meaning of each column:")
    st.text('Age : Age of the patient')
    st.text('Sex : Sex of the patient, 0:Male; 1:Female')
    st.text('''cp : Chest Pain type
      Value 1: typical angina
      Value 2: atypical angina
      Value 3: non-anginal pain
      Value 4: asymptomatic''')
    st.text('trtbps : resting blood pressure (in mm Hg)')
    st.text('''restecg: resting electrocardiographic results
      Value 0: normal
      Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
      Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    ''')
    st.text('thalach : maximum heart rate achieved')
    st.text('exang: exercise induced angina (1 = yes; 0 = no)')
    st.text('oldpeak')
    st.text('''slp:
        0-upsloping, 1-flat, 2-downsloping''')
    st.text('''caa:
        number of major vessels (0-4)''')
    st.text('thall: (0-4)')
    st.text('')
    st.markdown(html_temp2,unsafe_allow_html=True)
    st.text('')

    age = st.slider("age",0,100)
    sex = st.selectbox("sex",[0,1])
    cp = st.selectbox("cp",[0,1,2,3])
    trtbps = st.text_input("trtbps","Type Here")
    restecg = st.selectbox("restecg",[0,1,2])
    thalachh = st.text_input("thalachh","Type Here")
    exng = st.selectbox("exercise induced angina (1 = yes; 0 = no)",[0,1])
    oldpeak = st.text_input("oldpeak","Type Here")
    slp = st.selectbox("slp",[0,1,2])
    caa = st.selectbox("caa",[0,1,2,3,4])
    thall = st.selectbox("thall",[0,1,2,3])
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(age,sex,cp,trtbps,restecg,thalachh,exng,oldpeak,slp,caa,thall)
    st.success('The output is {}'.format(result))
    st.text('''
    0: less chance of heart attack
    1: more chance of heart attack
    ''')
    # if st.button("About"):
    #     st.text("Lets Learn")
    #     st.text("Built with Streamlit")
    
    df = pd.read_csv('heart.csv')
    st.subheader('Data Used:')
    st.dataframe(df)
if __name__=='__main__':
    main()