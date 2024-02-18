# books_words_extraction
import cProfile
from gensim.models import TfidfModel
import csv
import sys
import os
from PyQt5.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal, QThreadPool
from PyQt5.QtWidgets import QApplication, QMainWindow,QTableView, QTextEdit, QPushButton, QFileDialog, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget
from PyQt5.QtCore import QAbstractTableModel,Qt, QModelIndex
from PySide6.QtCore import QAbstractTableModel
import pdfplumber
import ebooklib
from ebooklib import epub
from reportlab.pdfgen import canvas
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
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import *
import re
from PyQt5 import QtCore
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            import traceback
            self.signals.error.emit((self.fn.__name__, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class TextEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Editor")
        self.setGeometry(100, 100, 500, 500)

        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(10, 10, 480, 400)

        self.load_button = QPushButton("Load", self)
        self.load_button.setGeometry(10, 420, 80, 30)
        self.load_button.clicked.connect(self.load_file)
        
        self.process_button = QPushButton("Process", self)
        self.process_button.setGeometry(100, 420, 80, 30)
        self.process_button.clicked.connect(self.process_text)

        self.save_button = QPushButton("Save", self)
        self.save_button.setGeometry(190, 420, 80, 30)
        self.save_button.clicked.connect(self.save_file)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.setGeometry(280, 420, 80, 30)
        self.clear_button.clicked.connect(self.clear_text)
        self.threadpool = QThreadPool() 

    def process_text(self):
        start_time = time.time()
        text = self.text_edit.toPlainText()
        #self.analyze_text(text)
    
        #existing preprocessing to get word_counts
        word_counts = self.analyze_text(text)  # Adjust analyze_text to return word_counts instead of setting the text directly
        if word_counts:  # To ensure this captures the return value
            self.translate_word_concurrently(word_counts)
        else:
            print("No word counts to process.")
        elapsed_time = time.time()
        print(f"Elapsed time: {elapsed_time - start_time} seconds")

    def analyze_text(self, text):
        try:
            
            
            
            def remove_special_tokens(tokens):
                
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
            
            # Tokenize the input text
            if not isinstance(text, str):
                raise ValueError("Input text must be a string")
            tokens = word_tokenize(text.lower())

            # Remove stop words
            stop_words = set(stopwords.words("english"))
            tokens = [token for token in tokens if token not in stop_words]

            
            # Remove numbers
            #tokens = [token for token in tokens if not token.isdigit()]
            tokens = [token for token in tokens if token.isalpha()]
            # Remove addresses
            
            
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

            # Remove names
            tagged_tokens = pos_tag(tokens)
            
            tags_to_exclude = ['NNP', 'NNPS','DT', 'CC','CD','EX','IN','JJS','LS','MD','POS','PRP','PRP$','RBR','SYM','UH','WDT', 'WP', 'WP$',  'TO']

            tokens = [token for token, pos in tagged_tokens if pos not in tags_to_exclude]
            # Lemmatize words
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            #####lemmatized = [lemmatizer.lemmatize(t) for t in no_stops]


            # Count the frequency of uncommon words
            #word_counts = self.word_count(tokens)
            word_counts = Counter(tokens)
            corpus=sent_tokenize(text)
            
            tfidf = TfidfModel(corpus=corpus)

            # # Calculate the tfidf weights of doc: tfidf_weights
            tfidf_weights = tfidf[tokens]

            # # Print the first five weights
            self.text_edit.setPlainText(str(tfidf_weights))
            #print(tfidf_weights)        
                    
            # Translate the words to Persian
            df = self.translate_counted_words(word_counts)
            
            self.text_edit.setPlainText(df.to_string(index=False))
            
            # Hide the text edit
            #self.text_edit.hide()
            self.process_button.setDisabled(True)
            #self.tableView.show()
            self.load_button.show()
            self.save_button.show()
            self.clear_button.show()
            self.clear_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.load_button.setEnabled(True)
            
            #self.save_table_data(self.tableView)
            
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except FileNotFoundError as fnf:
            print(f"FileNotFoundError: {fnf}")
        except Exception as e:
            print(f"An error occurred: {type(e).__name__} occured : {e}")
            
            self.text_edit.setPlainText(df.to_string(index=False))
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

    # def word_count(self, tokens):
    #     word_counts = {}
    #     for token in tokens:
    #         if token not in word_counts:
    #             word_counts[token] = 1
    #         else:
    #             word_counts[token] += 1
    #     return word_counts
            #return df
                
    def save_text_data(self, text_edit):
        with open('book_words.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            text = text_edit.toPlainText()
            writer.writerow([text])
        
    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
            
    def load_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;PDF Files (*.pdf);;EPUB Files (*.epub)")
        
        if file_path:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    self.text_edit.setPlainText(text)
            elif file_path.endswith(".epub"):
                book = epub.read_epub(file_path)
                text = ""
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        text += item.get_content().decode("utf-8")
                self.text_edit.setPlainText(text)
            else:
                with open(file_path, "r") as file:
                    self.text_edit.setPlainText(file.read())
        #self.analyze_text(text)
                
                
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

    def clear_text(self):
        if self.text_edit.toPlainText():
            self.save_file(self)
            self.text_edit.clear()
            self.tableView.clearSpans()
    
    def translate_word_concurrently(self, word_counts):
        for word, frequency in word_counts.items():
            worker = Worker(self.translate_and_update_ui, word=word, frequency=frequency)
            worker.signals.result.connect(self.handle_translation_result)
            self.threadpool.start(worker)

    def translate_and_update_ui(self, word, frequency):
        translation = GoogleTranslator(source='auto', target='fa').translate(word)
        return word, frequency, translation

    def handle_translation_result(self, result):
        word, frequency, translation = result

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = TextEditor()
    editor.show()
    sys.exit(app.exec_())
