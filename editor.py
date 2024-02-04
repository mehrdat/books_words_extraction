import csv
import sys
import cProfile

from PyQt5.QtWidgets import QApplication, QMainWindow,QTableView, QTextEdit, QPushButton, QFileDialog, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget
from PyQt5.QtCore import QAbstractTableModel,Qt, QModelIndex
from PySide6.QtCore import QAbstractTableModel

import traceback
import pdfplumber
import ebooklib
from ebooklib import epub
import reportlab
from reportlab.pdfgen import canvas
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tag import pos_tag
import PyDictionary
from PyDictionary import PyDictionary
import pandas as pd



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import *
import re
from PyQt5 import QtCore


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

    def process_text(self):
        text = self.text_edit.toPlainText()
        self.analyze_text(text)

    def analyze_text(self, text):
        try:
            def remove_special_tokens(tokens):
                #
                # Remove websites
                tokens = [token for token in tokens if not re.match(r'^https?:\/\/.*[\r\n]*', token)]
                # Remove numbers
                tokens = [token for token in tokens if not token.isdigit()]
                tokens = [token for token in tokens if token.isalpha()]
                # Remove addresses
                tokens = [token for token in tokens if not re.match(r'\d+.*\d+', token)]
                # Remove company names (replace with your own list of company names)
                company_names = ["Microsoft", "Google", "Apple"]
                tokens = [token for token in tokens if token not in company_names]
                
                #Remove special characters
                tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
                
                # Remove common words
                with open("/Users/Mehr/Desktop/Python/ed/10kwords.txt", "r") as file:
                    common_words = file.read().splitlines()
                tokens = [token for token in tokens if token.lower() not in common_words]

                return tokens
            
            # Tokenize the input text
            if not isinstance(text, str):
                raise ValueError("Input text must be a string")
            tokens = word_tokenize(text)

            # Remove stop words
            stop_words = set(stopwords.words("english"))
            tokens = [token for token in tokens if token.lower() not in stop_words]

            # Remove names
            tagged_tokens = pos_tag(tokens)
            
            tags_to_exclude = ['NNP', 'NNPS','DT', 'CC','CD','EX','IN','JJS','LS','MD','POS','PRP','PRP$','RBR','SYM','UH','WDT', 'WP', 'WP$',  'TO']
            
            def get_wordnet_pos(tag):
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return wordnet.NOUN

            tokens = [token for token, pos in tagged_tokens if pos not in tags_to_exclude]
            # Lemmatize words
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            #tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tag(tokens)]
            # Remove common words from open file
            tokens = remove_special_tokens(tokens)

            # Count the frequency of uncommon words
            word_counts = {}
            for token in tokens:
                if token not in word_counts:
                    word_counts[token] = 1
                else:
                    word_counts[token] += 1
            
            
            from google_trans_new import google_translator  
            translator = google_translator()  
            # Translate words using PyDictionary
            #dictionary = PyDictionary()
            #translations = {}
            #for word in word_counts.keys():
            #    translation = dictionary.meaning(word, disable_errors=True)
            #    if translation:
            #        translations[word] = translation\
            #from googletrans import Translator
            #translator = Translator(service_urls=['translate.googleapis.com'])
            #translations = {}
            #for word in word_counts.keys():
            #    result = translator.translate(word, src='en', dest='fa').text
            #    translation = translator.translate(word,lang_tgt='fa') 
            #    if translation:
            #        translations[word] = translation


            from deep_translator import GoogleTranslator
            
            translations = {}
            for word in word_counts.keys():
            
                translation=GoogleTranslator(source='auto', target='fa').translate(word)
                if translation:
                    translations[word] = translation
                    
                    
                    
            #    if result:
            #        translations[word] = result.text
                    
            #pip install google_trans_new

            #from google_trans_new import google_translator  
            #translator = google_translator()  
            #translate_text = translator.translate(word,lang_tgt='fa')  

            # Create a DataFrame with word, frequency, and meaning columns
            
            df = pd.DataFrame(columns=['word', 'frequency', 'meaning'])

            for word, frequency in word_counts.items():
                meaning = translations.get(word, '')
                df.loc[len(df)] = [word, frequency, meaning]
        
        

            # # Display the DataFrame in a QTableView
            # table_view = QTableView(self)
            # table_view.setGeometry(10, 10, 480, 400)
            # model = PandasModel(df)
            # table_view.setModel(model)
            # self.centralWidget = QWidget()
            # self.setCentralWidget(self.centralWidget)
            # self.layout = QVBoxLayout(self.centralWidget)
            # self.tableView = QTableView(self)

            # #self.model = PandasModel(dataframe)
            # self.tableView.setModel(self.model)
            # self.layout.addWidget(self.tableView)


            
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


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = TextEditor()
    editor.show()
    sys.exit(app.exec_())