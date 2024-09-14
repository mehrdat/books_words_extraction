from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QWidget,QVBoxLayout,QTextEdit
from PyQt5.QtWidgets import QPushButton
import pdfplumber
import ebooklib
from ebooklib import epub

import spacy 
from gensim.models import TfidfModel

import sys
import os
from PyQt5.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal, QThreadPool
from PyQt5.QtWidgets import QApplication, QMainWindow,QTableView, QTextEdit, QPushButton, QFileDialog, QVBoxLayout
from PyQt5.QtCore import QAbstractTableModel,Qt, QModelIndex
from gensim.utils import simple_preprocess

from ebooklib import epub
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from nltk.tag import pos_tag
import pandas as pd
import numpy as np


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk import *
import re
from PyQt5 import QtCore
import time
from collections import Counter

class EbookProc:
    def __init__(self,text):
        self.text=text
    
    def remove_special_tokens(self,tokens):
        # Remove websites
        tokens = [token for token in tokens if not re.match(r'^https?:\/\/.*[\r\n]*', token)]
        # Remove numbers
        tokens = [token for token in tokens if not token.isdigit()]
        tokens = [token for token in tokens if token.isalpha()]
        # Remove addresses
        tokens = [token for token in tokens if not re.match(r'\d+.*\d+', token)]
        
        #Remove special characters
        tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
        
        # Remove common words
        # # Get the directory of the current script
        current_dir = os.path.dirname(__file__)
        # # Build the path to the resource file
        resource_path = os.path.join(current_dir, 'resources', '10kwords.txt')
        with open(resource_path, "r") as file:
            common_words = file.read().splitlines()
        tokens = [token for token in tokens if token.lower() not in common_words]

        return tokens

    def process_text(self):
        start_time = time.time()
        
        word_counts = self.clean()
        print(word_counts)
        if word_counts:
            self.df=self.translate_counted_words(word_counts)
        else:
            print("No word counts to process.")
        elapsed_time = time.time()
        print(f"Elapsed time: {elapsed_time - start_time} seconds")
        
        return self.df

    def clean(self):
        try:
            #tokens = word_tokenize(text)
            tokens=simple_preprocess(self.text.lower())
            
            stop_words = set(stopwords.words("english"))
            tokens = [token for token in tokens if token not in stop_words]

            #tokens = [token for token in tokens if not token.isdigit()]
            tokens = [token for token in tokens if token.isalpha()]

            tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

            current_dir = os.path.dirname(__file__)
            resource_path = os.path.join(current_dir, 'resources', '10kwords.txt')
            with open(resource_path, "r") as file:
                common_words = file.read().splitlines()
            tokens = [token for token in tokens if token.lower() not in common_words]

            tagged_tokens = pos_tag(tokens)
            tags_to_exclude = ['NNP', 'NNPS','DT', 'CC','CD','EX','IN','JJS','LS','MD','POS','PRP','PRP$','RBR','SYM','UH','WDT', 'WP', 'WP$',  'TO']
            tokens = [token for token, pos in tagged_tokens if pos not in tags_to_exclude]
            
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]            
            
            #word_counts = self.word_count(tokens)
            word_counts = Counter(tokens)
            # corpus=sent_tokenize(text)
            
            # tfidf = TfidfModel(corpus=corpus)

            # # # Calculate the tfidf weights of doc: tfidf_weights
            # tfidf_weights = tfidf[tokens]

            # # Print the first five weights
            #self.text_edit.setPlainText(str(tfidf_weights))
            #print(tfidf_weights)        
                    
            # Translate the words to Persian
            #df = self.translate_counted_words(word_counts)
            
            #df=df.sort_values(by='frequency', ascending=False)
            #self.save_table_data(self.tableView)
            
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except FileNotFoundError as fnf:
            print(f"FileNotFoundError: {fnf}")
        except Exception as e:
            print(f"An error occurred: {type(e).__name__} occured : {e}")
            
        return word_counts

    def translate_counted_words(self, word_counts):
        translations = {}
        for word in word_counts.keys():
            translation=GoogleTranslator(source='auto', target='fa').translate(word)
            if translation:
                translations[word] = translation
                
        df = pd.DataFrame(columns=['word', 'frequency', 'meaning'])
        for word, frequency in word_counts.items():
            meaning = translations.get(word, '')
            df.loc[len(df)] = [word, frequency, meaning]
        return df

    def save_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;PDF Files (*.pdf);;EPUB Files (*.epub)")
        if file_path:
            if file_path.endswith(".pdf"):
                # Save as PDF
                c = canvas.Canvas(file_path)
                text = self.text_edit.toPlainText()
                c.drawString(100, 100, text)
                c.save()
                
            elif file_path.endswith(".epub"):
                # Save as EPUB
                # Implement your code here
                pass
            else:
                with open(file_path, "w") as file:
                    file.write(self.text_edit.toPlainText())


import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

class TextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
    def analyser(self,text):
        doc=self.nlp(text)
        entities=[(ent.text,ent.label_) for ent in doc.ents]
        tfidf_score=self._compute_tfidf(text)
        return {"entities":entities,"keywords":tfidf_score}
    
    def _compute_tfidf(self, text):
        vectorizer=TfidfVectorizer(max_features=100)
        vectors=vectorizer.fit_transform([text])
        return dict(zip(vectorizer.get_feature_names_out(),vectors.toarray().flatten()))
        
    def summarise(self,text):
        pass

    def create_embedding(self,text):
        embeddings=self.model.encode(text,convert_to_tensor=True)
        return embeddings


class VecDB:
    def __init__(self,dimension=768):
        self.index=faiss.IndexFlatL2(dimension)

    def add_vector(self,vectors):
        self.index.add(np.array(vectors))

    def search(self,query_vector,k=5):
        D,I=self.index.search(np.array([query_vector]),k)
        return I

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_gui()
        #self.proc=EbookProc()
        self.database=VecDB()
        self.analyzer=TextAnalyzer()
        
    def setup_gui(self):
        #self.setup_ui()
        self.setWindowTitle("Ebook Processor")
        self.setGeometry(300, 300, 600, 400)

        self.create_menu()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        self.text_area = QTextEdit()
        layout.addWidget(self.text_area)
        central_widget.setLayout(layout)

        self.button=QPushButton("Process")
        layout.addWidget(self.button)
        self.button.clicked.connect(self.process_text)
        central_widget.setLayout(layout)

        #self.button_proc=QPushButton("clear")
        #layout.addWidget(self.button_proc)
        #self.button.clicked.connect(self.proc)
        #central_widget.setLayout(layout)
    def create_menu(self):
        # Create a menu bar
        menu_bar = self.menuBar()

        # Create file menu and add actions
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.triggered.connect(self.open_file)

    def open_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open File", "", "All Files (*.*);;Text Files (*.txt);;PDF Files (*.pdf);;EPUB Files (*.epub)")        
        if file_path:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                    self.text_area.setPlainText(text)
                    
            elif file_path.endswith(".txt"):
                with open(file_path,"r") as f:
                    text=f.read()
                    
            elif file_path.endswith(".epub"):
                book = epub.read_epub(file_path)
                text=""
                for item in book.get_items():
                    if item.get_type==ebooklib.ITEM_DOCUMENT:
                        text+=item.get_body().decode("utf-8")
                        self.text_area.setPlainText(text)
        self.text_area.setPlainText(text)
        self.proc= EbookProc(text)
        
    def process_text(self):
        
        if self.proc:
            data = self.proc.process_text()  # Call the clean() method
            self.text_area.append("\nProcessed Data:\n" + str(data))
        else:
            self.text_area.append("\nNo text loaded. Please load a file first.")
            
        self.text_area.setPlainText(str(data))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
