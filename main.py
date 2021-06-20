import speech_recognition as sr
from time import ctime
import webbrowser
import time
import os
import playsound
import random
from gtts import gTTS
import pandas as pd
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim import corpora
import pprint
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

r = sr.Recognizer()

def record_audio():
    with sr.Microphone() as source:
        audio = r.listen(source)
        voice_data = " "
        try:
            voice_data = r.recognize_google(audio)
            question_orig = voice_data
            question = clean_sentence(question_orig, stopwords=False)
            print(question)
        except sr.UnknownValueError:
            print("Please come again")
        except sr.RequestError:
            print("Sorry the system is down")

        return question

time.sleep(1)
print("talk to the mic, im listening")
while 1:
    voice_data = record_audio()


df = pd.read_csv("/Users/applecare/Downloads/QuestionAnswering_From_FAQ_Tutorial-master/FAQ_MachineLearningInterview_com.csv")
df.columns = ["questions", "answers"]


def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)  # remove all non alpha-numeric xters

    if stopwords:
        sentence = remove_stopwords(sentence)

    return sentence


def get_cleaned_sentences(df, stopwords=False):
    sents = df[["questions"]]
    cleaned_sentences = []

    for index, row in df.iterrows():
        cleaned = clean_sentence(row["questions"], stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
cleaned_sentences_with_stopwords = get_cleaned_sentences(df,stopwords=False)

def retrieve_and_print_faq_answer(question_embedding,sentence_embeddings, FAQdf, sentences):
    max_sim = -1
    index_sim = -1
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding,question_embedding)[0][0]
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    print(FAQdf.iloc[index_sim, 1])


with sr.Microphone() as source:
    audio = r.listen(source)
voice_data = r.recognize_google(audio)
question_orig = voice_data
question = clean_sentence(question_orig, stopwords=False)
sentences = cleaned_sentences_with_stopwords
sentence_words = [[word for word in document.split()]
                  for document in sentences]
dictionary = corpora.Dictionary(sentence_words)
bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
question_embedding = dictionary.doc2bow(question.split())

retrieve_and_print_faq_answer(question_embedding, bow_corpus, df, sentences)




'''def alexis_speak(audio_string):
    tts = gTTS(text=audio_string, lang="en")
    r = random.randint(1,1000000)
    audio_file = "audio-" + str(r) + ".mp3"
    tts.save(audio_file)
    playsound.playsound(audio_file)
    print(audio_string)
    os.remove(audio_file)'''








