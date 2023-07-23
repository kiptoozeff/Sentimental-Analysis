import streamlit as st
import streamlit.components.v1 as com
#import libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
#convert logits to probabilities
from scipy.special import softmax
from transformers import pipeline

#Set the page configs
st.set_page_config(page_title='Sentiments Analysis',page_icon='😎',layout='wide')

#welcome Animation
com.iframe("https://embed.lottiefiles.com/animation/149093")
st.markdown("<h1 style='text-align: center'> Covid Vaccine Tweet Sentiments </h1>",unsafe_allow_html=True)
st.write("<h2 style='font-size: 24px;'> These models were trained to detect how a user feels about the covid vaccines based on their tweets(text) </h2>",unsafe_allow_html=True)

#Create a form to take user inputs
with st.form(key='tweet',clear_on_submit=True):
    #input text
    text=st.text_area('Copy and paste a tweet or type one',placeholder='I find it quite amusing how people ignore the effects of not taking the vaccine')
    #Set examples
    alt_text=st.selectbox("Can't Type?  Select an Example below",('I hate the vaccines','Vaccines made from dead human tissues','Take the vaccines or regret the consequences','Covid is a Hoax','Making the vaccines is a huge step forward for humanity. Just take them'))
    #Select a model
    models={'Bert':'penscola/tweet_sentiments_analysis_bert',
            'Distilbert':'penscola/tweet_sentiments_analysis_distilbert',
            'Roberta':'penscola/tweet_sentiments_analysis_roberta'}
    model=st.selectbox('Which model would you want to Use?',('Bert','Distilbert','Roberta'))
     #Submit
    submit=st.form_submit_button('Predict','Continue processing input')
    
selected_model=models[model]


#create columns to show outputs
col1,col2,col3=st.columns(3)
col1.write('<h2 style="font-size: 24px;"> Sentiment Emoji </h2>',unsafe_allow_html=True)
col2.write('<h2 style="font-size: 24px;"> How this user feels about the vaccine </h2>',unsafe_allow_html=True)
col3.write('<h2 style="font-size: 24px;"> Confidence of this prediction </h2>',unsafe_allow_html=True)

if submit:
    #Check text
    if text=="":
        text=alt_text
        st.success(f"input text is set to '{text}'")    
    else:
        st.success('Text received',icon='✅')
        
    #import the model
    pipe=pipeline(model=selected_model)

#pass text to model
    output=pipe(text)
    output_dict=output[0]
    lable=output_dict['label']
    score=output_dict['score']
    
        #output
    if lable=='NEGATIVE' or lable=='LABEL_0':
        with col1:
            com.iframe("https://embed.lottiefiles.com/animation/125694")
        col2.write('NEGATIVE')
        col3.write(f'{score:.2%}')
    elif lable=='POSITIVE'or lable=='LABEL_2':
        with col1:
            com.iframe("https://embed.lottiefiles.com/animation/148485")
        col2.write('POSITIVE')
        col3.write(f'{score:.2%}')
    else:
        with col1:
            com.iframe("https://embed.lottiefiles.com/animation/136052")
        col2.write('NEUTRAL')
        col3.write(f'{score:.2%}')