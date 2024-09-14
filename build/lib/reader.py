import sys
from deep_translator import GoogleTranslator
import spacy
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog
import cProfile
import pstats
import spacy 
from gensim.models import TfidfModel
import csv
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer

from PySide6.QtCore import QAbstractTableModel
import pdfplumber
import ebooklib
from ebooklib import epub

import timeit
import pandas as pd

from nltk import *
import re

import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import nltk
from googletrans import Translator


class MainWindow(QMainWindow):
    def __init__(self):
    
        super().__init__()
        self.lemma=[]
        self.text=""
        self.cleaned=[]
        self.cleaned_10k=[]
        self.tf_idf=pd.DataFrame()
        self.counted_words=Counter()
        self.words=pd.DataFrame()
        self.translations=[]
        self.setWindowTitle("Reader")
        self.setGeometry(200, 175, 800, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_btn)

        self.proc_alpha_lemma = QPushButton("alpha lemma")
        self.proc_alpha_lemma.clicked.connect(self.lower_alpha_lemma)
        self.layout.addWidget(self.proc_alpha_lemma)
        
        self.proc_10k_removal = QPushButton("10 K removal")
        self.proc_10k_removal.clicked.connect(self.remove_common_words)
        self.layout.addWidget(self.proc_10k_removal)
        
        self.proc_tfidf=QPushButton("TF-IDF")
        self.proc_tfidf.clicked.connect(self.tfidf)
        self.layout.addWidget(self.proc_tfidf)
        
        self.proc_translate=QPushButton("Translate")
        self.proc_translate.clicked.connect(self.translate)
        self.layout.addWidget(self.proc_translate)
        
        self.text_area = QTextEdit()
        self.layout.addWidget(self.text_area)
        #self.text_area.setWordWrapMode(0)

    def load_file(self):
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open File", "", "PDF Files (*.pdf);;Text Files (*.txt);;EPUB Files (*.epub)")
        
        if file_path:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    self.text_area.setPlainText(text)
            elif file_path.endswith(".epub"):
                book = epub.read_epub(file_path)
                text = ""
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        text += item.get_content().decode("utf-8")
                self.text_area.setPlainText(text)
            else:
                with open(file_path, "r") as file:
                    self.text_area.setPlainText(file.read())
                    self.text=file.read()

    def lower_alpha_lemma(self):
        """ returns the text in lower case and the lemma of the text
        removes digits and spaces and most characters as well as stop words"""
        # import the text from the text area
        self.text = self.text_area.toPlainText()
        
        #import the text into the spacy model
        nlp=spacy.load('en_core_web_sm')
        
        
        #lower the text
        self.text = self.text.lower()
        
        doc=nlp(self.text)
        stopwords=spacy.lang.en.stop_words.STOP_WORDS
        
        
        #extract the lemma of the text
        self.lemma=[token.lemma_ for token in doc if token.is_stop==False and token.is_punct==False and token.is_space==False and token.is_alpha==True  not in stopwords and len(token)>2 and not token.pos_ in ["AUX", "DET", "ADP", "CCONJ", "SCONJ", "PART","PROPN","NOUN"] and self.has_meaning(token) ]
        #cleaned = " ".join([token.text for token in doc if not token.is_stop and not token.pos_ in ["AUX", "DET", "ADP", "CCONJ", "SCONJ", "PART"]])
        
        #clean the text from non-alphabetic characters
        #self.cleaned= ' '.join([lem for lem in self.lemma if  lem not in stopwords and len(lem)>2])
        self.cleaned=' '.join(self.lemma)
        
        self.text_area.clear()
        print(self.cleaned)
        self.text_area.append(self.cleaned)
    
    def remove_common_words(self):
        """ returns the text without the 10k most common words"""
        current_dir = os.path.dirname(__file__)
            # # Build the path to the resource file
        resource_path = os.path.join(current_dir, 'resources', '10kwords.txt')
        with open(resource_path, "r") as file:
            common_words = file.read().splitlines()
        self.cleaned_10k = [token for token in self.lemma if token not in common_words]
        
        temp=' '.join(self.cleaned_10k)
        self.text_area.clear()
        print(self.cleaned_10k)
        self.text_area.append(temp)
    
    def tfidf(self):
        """ returns the tfidf of the text and
        chooses the most imporatnt words in the given text"""
            # Add your code here
        tf=TfidfVectorizer()
        vector=tf.fit_transform([' '.join(self.cleaned_10k)])
        
        self.tf_idf=pd.DataFrame(vector.toarray(),columns=tf.get_feature_names_out())
        self.tf_idf = self.tf_idf.transpose().reset_index().rename(columns={'index': 'word'}).iloc[1:].reset_index(drop=True).rename(columns={0: 'TF_IDF'})

        #self.tf_idf = pd.DataFrame({'word': tf.get_feature_names_out(), 'weight': tf.iloc[:, ::-1].sort_values(by=0, ascending=False)[0].values})
        #self.tf_idf=self.tf_idf.T
        #self.tf_idf=pd.DataFrame({word: tf.idf_[tf.vocabulary_.get(word)] for word in tf.get_feature_names_out()})
        
        #self.tf_idf=pd.DataFrame({'vecotr' :vector.toarray(),'word':tf.get_feature_names_out() } )
        
        #self.tf_idf=self.tf_idf.T
        #self.tf_idf = self.tf_idf.iloc[:, ::-1]

        #self.tf_idf=self.tf_idf.sum().sort_values(ascending=False)
        #self.tf_idf.columns = ['word', 'weight']
        #self.tf_idf.to_csv("tf.csv")
        self.text_area.clear()
        print(self.tf_idf)
        self.text_area.append(self.tf_idf.to_string())
    
    def count_word(self):
        """ returns the count of the words in the text"""
        # Add your code here
        self.counted_words=Counter(self.text.split())
    def translate(self):
        """ returns the translation of the text"""
        #self.count_word(self)
        
        #self.counted_words=Counter(self.tf_idf.iloc[:,0])
        #print(self.counted_words[:20])
        
        # Using googletrans library
        translator = Translator(service_urls=['translate.google.com'])
        # translations_googletrans = [translator.translate(word, dest='fa').text for word,_ in t.items()]
        
        def translate_word(word):
            translation = translator.translate(word, dest='fa')
            return translation.text

# Apply translation to each word in the dataframe
        self.tf_idf['translated_word'] = self.tf_idf['word'].apply(translate_word)

    
        # we need to translate the tfidf words
        #translations_googletrans=translations = {word: translator.translate(word, dest='fa').text for word,_ in self.counted_words.most_common() }
        
        
        
        #translations_googletrans=translations = {word: translator.translate(word, dest='fa').text for word,_ in self.counted_words.most_common() }
        
        # self.translations = pd.DataFrame(columns=['word', 'frequency', 'meaning'])
        # for word, frequency in self.counted_words.most_common():
        #     meaning = translations.get(word, '')
        #     self.translations.loc[len(self.translations)] = [word, frequency, meaning]
        
        
        self.text_area.clear()
        print(self.tf_idf['translated_word'].head(20))
        self.text_area.append(self.tf_idf.to_string())
        
        
    def has_meaning(self,token):
        """
        Checks if a word likely has meaning, combining spaCy and external resources.
        """

        # spaCy checks
        if token.pos_ in ["UH", "SYM"]:
            return False
        if not token.has_vector:
            return False

        # Dictionary lookup
        # if nltk.wordnet.synsets(token):
        #     return True
        
        return True
        
if __name__ == "__main__":
    with cProfile.Profile() as pr:
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        
        # elapsed_time = timeit.timeit(window.lower_alpha_lemma, number=5)
        # print(f"my_method execution time: {elapsed_time:.4f} seconds")
        
        # elapsed_time = timeit.timeit(window.remove_common_words, number=5)
        # print(f"my_method execution time: {elapsed_time:.4f} seconds")
        
        res=pstats.Stats(pr)
        res.sort_stats(pstats.SortKey.TIME)
        res.print_stats()
        sys.exit(app.exec())
        
        

