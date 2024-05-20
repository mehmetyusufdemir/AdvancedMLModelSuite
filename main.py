import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from itertools import cycle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, log_loss, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImPipeline
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QMessageBox, QFileDialog, QInputDialog, QTableWidget, QTableWidgetItem, QTextEdit, QComboBox, QListWidget, QAbstractItemView, QDialog, QFormLayout, QLineEdit, QAction, QMenu
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QThread
from PyQt5.QtGui import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class CatBoostParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CatBoost Parametreleri")
        self.layout = QVBoxLayout(self)

        # Parametreler için form oluştur
        formLayout = QFormLayout()
        self.learning_rate_input = QLineEdit("0.001")
        self.depth_input = QLineEdit("8")
        self.iterations_input = QLineEdit("100")
        self.early_stopping_rounds_input = QLineEdit("20")
        self.val_size_input = QLineEdit("0.1")
        self.test_size_input = QLineEdit("0.2")
        self.batch_size_input = QLineEdit("32")
        self.border_count_input = QLineEdit("254")
        self.l2_leaf_reg_input = QLineEdit("7")
        self.bagging_temperature_input = QLineEdit("5")

        formLayout.addRow(QLabel("Learning Rate:"), self.learning_rate_input)
        formLayout.addRow(QLabel("Depth:"), self.depth_input)
        formLayout.addRow(QLabel("Iterations:"), self.iterations_input)
        formLayout.addRow(QLabel("Early Stopping Rounds:"), self.early_stopping_rounds_input)
        formLayout.addRow(QLabel("Validation Set Size:"), self.val_size_input)
        formLayout.addRow(QLabel("Test Set Size:"), self.test_size_input)
        formLayout.addRow(QLabel("Batch Size:"), self.batch_size_input)
        formLayout.addRow(QLabel("Border Count:"), self.border_count_input)
        formLayout.addRow(QLabel("L2 Leaf Regularization:"), self.l2_leaf_reg_input)
        formLayout.addRow(QLabel("Bagging Temperature:"), self.bagging_temperature_input)

        self.layout.addLayout(formLayout)

        # Onay butonu
        okButton = QPushButton("Tamam")
        okButton.clicked.connect(self.accept)
        self.layout.addWidget(okButton)

    def get_parameters(self):
        return {
            "learning_rate": float(self.learning_rate_input.text()),
            "depth": int(self.depth_input.text()),
            "iterations": int(self.iterations_input.text()),
            "early_stopping_rounds": int(self.early_stopping_rounds_input.text()),
            "val_size": float(self.val_size_input.text()),
            "test_size": float(self.test_size_input.text()),
            "batch_size": int(self.batch_size_input.text()),
            "border_count": int(self.border_count_input.text()),
            "l2_leaf_reg": float(self.l2_leaf_reg_input.text()),
            "bagging_temperature": float(self.bagging_temperature_input.text())
        }

class XGBoostParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("XGBoost Parametreleri")
        self.layout = QVBoxLayout(self)

        # Parametreler için form oluştur
        formLayout = QFormLayout()
        self.n_estimators_input = QLineEdit("100")
        self.max_depth_input = QLineEdit("6")
        self.min_child_weight_input = QLineEdit("1")
        self.gamma_input = QLineEdit("0")
        self.subsample_input = QLineEdit("0.8")
        self.colsample_bytree_input = QLineEdit("0.8")
        self.reg_alpha_input = QLineEdit("0.01")
        self.reg_lambda_input = QLineEdit("1")
        self.learning_rate_input=QLineEdit("0.001")
        self.early_stopping_rounds_input = QLineEdit("10")
        self.val_size_input = QLineEdit("0.1")
        self.test_size_input = QLineEdit("0.2")

        formLayout.addRow(QLabel("n_estimators:"), self.n_estimators_input)
        formLayout.addRow(QLabel("max_depth:"), self.max_depth_input)
        formLayout.addRow(QLabel("min_child_weight:"), self.min_child_weight_input)
        formLayout.addRow(QLabel("gamma:"), self.gamma_input)
        formLayout.addRow(QLabel("subsample:"), self.subsample_input)
        formLayout.addRow(QLabel("colsample_bytree:"), self.colsample_bytree_input)
        formLayout.addRow(QLabel("reg_alpha (L1 regularization):"), self.reg_alpha_input)
        formLayout.addRow(QLabel("reg_lambda (L2 regularization):"), self.reg_lambda_input)
        formLayout.addRow(QLabel("learning_rate:"), self.learning_rate_input)
        formLayout.addRow(QLabel("early_stopping_rounds:"), self.early_stopping_rounds_input)
        formLayout.addRow(QLabel("Validation set size (e.g., 0.1):"), self.val_size_input)
        formLayout.addRow(QLabel("Test set size (e.g., 0.2):"), self.test_size_input)

        self.layout.addLayout(formLayout)

        # Onay butonu
        okButton = QPushButton("Tamam")
        okButton.clicked.connect(self.accept)
        self.layout.addWidget(okButton)

    def get_parameters(self):
        return {
            "n_estimators": int(self.n_estimators_input.text()),
            "max_depth": int(self.max_depth_input.text()),
            "min_child_weight": float(self.min_child_weight_input.text()),
            "gamma": float(self.gamma_input.text()),
            "subsample": float(self.subsample_input.text()),
            "colsample_bytree": float(self.colsample_bytree_input.text()),
            "reg_alpha": float(self.reg_alpha_input.text()),
            "reg_lambda": float(self.reg_lambda_input.text()),
            "learning_rate": float(self.learning_rate_input.text()),
            "early_stopping_rounds": int(self.early_stopping_rounds_input.text()),
            "val_size": float(self.val_size_input.text()),
            "test_size": float(self.test_size_input.text())
        }

class NaiveBayesParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Naive Bayes Parametreleri")
        self.layout = QVBoxLayout(self)

        formLayout = QFormLayout()

        # Alpha parametresi için input
        self.alpha_input = QLineEdit("1.0")
        formLayout.addRow(QLabel("Alpha:"), self.alpha_input)

        # Fit_prior parametresi için combobox
        self.fit_prior_input = QComboBox()
        self.fit_prior_input.addItems(["True", "False"])
        formLayout.addRow(QLabel("Fit Prior:"), self.fit_prior_input)

        # Test size ve validation size parametreleri için input
        self.test_size_input = QLineEdit("0.2")
        self.validation_size_input = QLineEdit("0.1")
        formLayout.addRow(QLabel("Test Size (e.g., 0.2):"), self.test_size_input)
        formLayout.addRow(QLabel("Validation Size (e.g., 0.1):"), self.validation_size_input)

        self.layout.addLayout(formLayout)

        # Onay butonu
        okButton = QPushButton("Tamam")
        okButton.clicked.connect(self.accept)
        self.layout.addWidget(okButton)

    def get_parameters(self):
        return {
            "alpha": float(self.alpha_input.text()),
            "fit_prior": self.fit_prior_input.currentText() == "True",
            "test_size": float(self.test_size_input.text()),
            "validation_size": float(self.validation_size_input.text())
        }

class KNNParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Parametreleri")
        self.layout = QVBoxLayout(self)

        formLayout = QFormLayout()

        # KNN için parametreler
        self.n_neighbors_input = QLineEdit("5")
        self.weights_input = QComboBox()
        self.weights_input.addItems(["uniform", "distance"])
        self.metric_input = QComboBox()
        self.metric_input.addItems(["minkowski", "euclidean", "manhattan"])

        # Derin öğrenme veya diğer modeller için genel parametreler
        self.test_size_input = QLineEdit("0.2")
        self.val_size_input = QLineEdit("0.1")
        self.batch_size_input = QLineEdit("32")
        self.epochs_input = QLineEdit("100")

        # KNN parametrelerini form layout'a ekle
        formLayout.addRow(QLabel("n_neighbors:"), self.n_neighbors_input)
        formLayout.addRow(QLabel("weights:"), self.weights_input)
        formLayout.addRow(QLabel("metric:"), self.metric_input)

        # Genel parametreleri form layout'a ekle
        formLayout.addRow(QLabel("Test Seti Boyutu:"), self.test_size_input)
        formLayout.addRow(QLabel("Doğrulama Seti Boyutu:"), self.val_size_input)
        formLayout.addRow(QLabel("Batch Size:"), self.batch_size_input)
        formLayout.addRow(QLabel("Epoch Sayısı:"), self.epochs_input)

        self.layout.addLayout(formLayout)
        okButton = QPushButton("Tamam")
        okButton.clicked.connect(self.accept)
        self.layout.addWidget(okButton)

    def get_parameters(self):
        return {
            "n_neighbors": int(self.n_neighbors_input.text()),
            "weights": self.weights_input.currentText(),
            "metric": self.metric_input.currentText(),
            "test_size": float(self.test_size_input.text()),
            "val_size": float(self.val_size_input.text()),
            "batch_size": int(self.batch_size_input.text()),
            "epochs": int(self.epochs_input.text())
        }

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yardım")
        self.showMaximized()  # Diyalog penceresini tam ekran yap
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)  # Tam ekran butonunu etkinleştir

        layout = QVBoxLayout(self)

        # Açıklamaları içeren metin alanı
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <p><strong><span style="color: black; font-size: 16pt;">Dosya Seç:</span></strong> <span style="font-size: 14pt;">Veri setini yüklemek için kullanılır.Veriseti csv formatında olmalıdır</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">İstatistikler:</span></strong> <span style="font-size: 14pt;">Veri seti üzerinde kullanıcının temel istatistiksel analizler yapar.</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Feature Generation:</span></strong> <span style="font-size: 14pt;">Veri setinden yeni öznitelikler türetmek ve silmek  için kullanılır.</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Boşluk Doldurma:</span></strong> <span style="font-size: 14pt;">Eksik verileri doldurmak için çeşitli yöntemler sunar.Her boş sütun için ayrı ayrı işlemler yapılır ve boşluklar doldurulduktan sonra kullanıcı yeni oluşan dosyayı kaydetmek zorundadır.</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Tarih İşlemleri:</span></strong> <span style="font-size: 14pt;">Verisetinden kullanıcının istediği tarih bilgisini çekip bu veri ile yeni bir öznitelik oluşturmaya yarar.Kullanıcı tarih işleminde kullanılan karakter bilgisini ekrana girer</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Model Eğit:</span></strong> <span style="font-size: 14pt;">Seçilen algoritma ile model eğitimi yapar.Hiperparametreleri kullanıcı girmelidir.Modeller çalıştırılmadan önce boşluk doldurma işlemleri yapılmalıdır</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Grafik İşlemleri:</span></strong> <span style="font-size: 14pt;">Veri seti üzerinden grafikler oluşturur.</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Feature Selection:</span></strong> <span style="font-size: 14pt;">Öznitelik seçimi yapar.Verisetini minimalize eder.</span></p>
        <p><strong><span style="color: black; font-size: 16pt;">Explain AI:</span></strong> <span style="font-size: 14pt;">Model açıklamaları için LIME ve SHAP gibi yöntemler sunar.SHAP, makine öğrenimi modellerinin kararlarını açıklamak için özellik önemini hesaplar.Lime ise yerel tahminler yaparak tahminlerin arkasındaki mantığı açıklamaya yarar.</span></p>
        """)
        layout.addWidget(help_text)
        self.finished.connect(parent.help_dialog_closed)


        # Diyalogu kapatma butonu
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def toggle_help_dialog(self):
        if self.is_help_dialog_open:
            self.help_dialog.close()
        else:
            self.help_dialog.show()
            self.is_help_dialog_open = True
            self.help_button.setStyleSheet(self.active_style)  # Butona aktif stil uygula

    def help_dialog_closed(self):
        self.is_help_dialog_open = False
        self.help_button.setStyleSheet(self.button_style_menu)  # Butona normal stil uygula

class RandomForestParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random Forest Parametreleri")
        self.layout = QVBoxLayout(self)

        # Parametreler için form oluştur
        formLayout = QFormLayout()
        self.n_estimators_input = QLineEdit("100")
        self.max_depth_input = QLineEdit("None")
        self.min_samples_split_input = QLineEdit("2")
        self.min_samples_leaf_input = QLineEdit("1")
        self.bootstrap_input = QComboBox()
        self.bootstrap_input.addItems(["True", "False"])
        self.max_features_input=QComboBox()
        self.max_features_input.addItems(["None","sqrt","log2"])
        self.max_samples_input = QLineEdit("None")  # Kullanıcıdan oran veya örnek sayısı olarak giriş yapmasını bekleyin
        self.oob_score_input = QComboBox()
        self.oob_score_input.addItems(["True", "False"])

        formLayout.addRow(QLabel("n_estimators:"), self.n_estimators_input)
        formLayout.addRow(QLabel("max_depth (None için boş bırakın):"), self.max_depth_input)
        formLayout.addRow(QLabel("min_samples_split:"), self.min_samples_split_input)
        formLayout.addRow(QLabel("min_samples_leaf:"), self.min_samples_leaf_input)
        formLayout.addRow(QLabel("bootstrap (True/False):"), self.bootstrap_input)
        formLayout.addRow(QLabel("max_features "), self.max_features_input)
        formLayout.addRow(QLabel("max_samples (Oran olarak 0.1 gibi veya örnek sayısı olarak 100 gibi girilebilir):"), self.max_samples_input)
        formLayout.addRow(QLabel("oob_score (True/False):"), self.oob_score_input)

        self.layout.addLayout(formLayout)

        # Onay butonu
        okButton = QPushButton("Tamam")
        okButton.clicked.connect(self.accept)
        self.layout.addWidget(okButton)

    def get_parameters(self):
        max_depth = self.max_depth_input.text()
        max_samples = self.max_samples_input.text()
        # max_samples değerini doğru şekilde işle
        max_samples = None if max_samples.lower() == "none" else float(
            max_samples) if '.' in max_samples or '0' <= max_samples <= '1' else int(max_samples)
        return {
            "n_estimators": int(self.n_estimators_input.text()),
            "max_depth": None if max_depth.lower() == "none" else int(max_depth),
            "min_samples_split": int(self.min_samples_split_input.text()),
            "min_samples_leaf": int(self.min_samples_leaf_input.text()),
            "bootstrap": self.bootstrap_input.currentText() == "True",
            "max_features": self.max_features_input.currentText() if self.max_features_input.currentText().lower() != "none" else None,
            "max_samples": max_samples,
            "oob_score": self.oob_score_input.currentText() == "True"
        }

class HistogramDialog(QDialog):
    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.figure = figure
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(self.figure)
        layout.addWidget(canvas)

        close_button = QPushButton("Kapat")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

class DateTimeAnalysisWidget(QWidget):
    def __init__(self, filename, output_text_widget):
        super().__init__()
        self.filename = filename
        self.output_text_widget = output_text_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Tarih/Timestamp Sütunu:"))
        self.datetime_column_combobox = QComboBox()
        layout.addWidget(self.datetime_column_combobox)

        layout.addWidget(QLabel("Zaman Birimi:"))
        self.unit_combobox = QComboBox()
        self.unit_combobox.addItems(["yıl", "ay", "gün"])
        layout.addWidget(self.unit_combobox)

        self.setLayout(layout)

        if self.filename:
            self.load_datetime_columns()

    def load_datetime_columns(self):
        df = pd.read_csv(self.filename, parse_dates=True, infer_datetime_format=True)
        datetime_columns = self.identify_datetime_columns(df)
        self.datetime_column_combobox.clear()
        self.datetime_column_combobox.addItems(datetime_columns)

    def identify_datetime_columns(self, df):
        datetime_columns = []
        charr, ok = QInputDialog.getText(self, 'Tarih formatı için karakter Gir', 'Kaldırılacak Karakter:')
        if ok and charr:
            for col in df.columns:
                # Tüm hücre değerlerini metin tipine çevir
                df[col] = df[col].astype(str)
                # Yalnızca hücrelerde aranan karakter varsa değiştir
                df[col] = df[col].apply(lambda x: x.replace(charr, '/') if charr in x else x)
                try:
                    # Veriyi datetime'a dönüştürmeyi deneyin, başarılı olursa bu sütunu ekleyin
                    pd.to_datetime(df[col], errors='coerce')
                    datetime_columns.append(col)
                except ValueError:
                    continue

        return datetime_columns

    def add_datetime_feature(self):
        column = self.datetime_column_combobox.currentText()
        unit = self.unit_combobox.currentText()

        df = pd.read_csv(self.filename)
        df[column] = pd.to_datetime(df[column], errors='coerce')

        if unit == 'yıl':
            df['year'] = df[column].dt.year
        elif unit == 'ay':
            df['month'] = df[column].dt.month
        elif unit == 'gün':
            df['day'] = df[column].dt.day
        else:
            self.output_text_widget.setPlainText("Bilinmeyen zaman birimi")
            return

        self.output_text_widget.setPlainText(f"{unit.capitalize()} bilgisi '{unit}' sütunu olarak eklendi.")

        # Yeni veri setini kaydet
        save_option = QMessageBox.question(self, "Kaydet", "Değişiklikleri yeni bir dosyada kaydetmek ister misiniz?",
                                           QMessageBox.Yes | QMessageBox.No)
        if save_option == QMessageBox.Yes:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Yeni Veri Seti Kaydet", "", "CSV Dosyaları (*.csv)",
                                                       options=options)
            if file_name:
                df.to_csv(file_name, index=False)
                self.output_text_widget.append(f"Değişiklikler başarıyla kaydedildi: {file_name}")

        else:
            result = "Bilinmeyen zaman birimi"

        formatted_result = '\n'.join([f"{index}: {count}" for index, count in result.items()])
        self.output_text_widget.setPlainText(f"{unit.capitalize()} bazında veri sayısı:\n{formatted_result}")

    def update_widget(self, filename):
        self.filename = filename
        self.load_datetime_columns()

class SVMDetailsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SVM Parametreleri")
        self.layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.c_input = QLineEdit("1.0")
        self.kernel_input = QComboBox()
        self.kernel_input.addItems(["linear", "poly", "rbf", "sigmoid"])
        self.degree_input = QLineEdit("3")  # Varsayılan olarak 3
        self.gamma_input = QLineEdit("scale")
        self.coef0_input = QLineEdit("0.0")  # Polinom ve sigmoid için kullanılır
        self.max_iter_input = QLineEdit("-1")
        self.test_size_input = QLineEdit("0.2")
        self.validation_size_input = QLineEdit("0.1")

        form_layout.addRow(QLabel("C (Düzenleme parametresi):"), self.c_input)
        form_layout.addRow(QLabel("Kernel:"), self.kernel_input)
        form_layout.addRow(QLabel("Degree (only for 'poly'):"), self.degree_input)
        form_layout.addRow(QLabel("Gamma: scale/auto/float"), self.gamma_input)
        form_layout.addRow(QLabel("Coef0 (for 'poly' and 'sigmoid'):"), self.coef0_input)
        form_layout.addRow(QLabel("Max Iterations (-1 for no limit):"), self.max_iter_input)
        form_layout.addRow(QLabel("Test Size (e.g., 0.2):"), self.test_size_input)
        form_layout.addRow(QLabel("Validation Size (e.g., 0.1):"), self.validation_size_input)

        self.layout.addLayout(form_layout)

        ok_button = QPushButton("Tamam")
        ok_button.clicked.connect(self.accept)
        self.layout.addWidget(ok_button)

    def get_parameters(self):
        return {
            "C": float(self.c_input.text()),
            "kernel": self.kernel_input.currentText(),
            "degree": int(self.degree_input.text()),
            "gamma": self.gamma_input.text(),
            "coef0": float(self.coef0_input.text()),
            "max_iter": int(self.max_iter_input.text()),
            "test_size": float(self.test_size_input.text()),
            "validation_size": float(self.validation_size_input.text())
        }

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.filename = None  # Başlangıçta filename'ı None olarak ayarla
        self.datetime_analysis_widget = None  # Initially set it to None
        # MainWindow sınıfının __init__ metoduna ekleyin
        self.cleaned_data_filename = None  # Boşluk doldurma sonrası oluşturulan dosyanın adını tutacak
        self.clf = None  # RandomForestClassifier nesnesini saklamak için
        self.X_train_columns = None  # Eğitim sırasında kullanılan sütun adlarını saklamak için

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Siber Saldırı Tespit Sistemi")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: darkturquoise; color: white;")
        self.main_layout = QHBoxLayout()  # main_layout'u burada tanımlayın

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Sol tarafta menüler için bir bölüm
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(0, 0, 0, 0)  # Sol, üst, sağ ve alt kenar boşluklarını kaldır

        self.button_style_menu = """
                           QPushButton {
                               background-color: #ff1493;
                               color: black;
                               border-style: solid;
                               border-width: 2px;
                               border-radius: 15px;
                               border-color: #ff1493;
                               padding: 10px 25px;
                               font-size: 18px;
                               font-weight: bold;
                           }
                           QPushButton:hover {
                               background-color: #ff69b4;
                               border-color: #ff69b4;
                           }
                           QPushButton:pressed {
                               background-color: #ff85c1;
                               border-color: #ff85c1;
                           }
                       """
        self.active_style = """
            QPushButton {
                background-color: #ff85c1;  /* Daha açık bir arka plan rengi */
                color: black;
                border-style: solid;
                border-width: 2px;
                border-radius: 15px;
                border-color: #ff85c1;
                padding: 10px 25px;
                font-size: 18px;
                font-weight: bold;
                opacity: 0.6; /* Bu özellik PyQt5'te işlevsel olmayabilir */
            }
            QPushButton:hover {
                background-color: #ffb3e6; /* Daha açık hover rengi */
                border-color: #ffb3e6;
            }
            QPushButton:pressed {
                background-color: #ffcce6; /* Daha açık pressed rengi */
                border-color: #ffcce6;
            }
        """
        self.file_button = QPushButton("Dosya Seç")
        self.file_button.setStyleSheet(self.button_style_menu)
        self.file_button.clicked.connect(self.select_file)
        left_layout.addWidget(self.file_button)

        self.statistics_button = QPushButton("İstatistikler")
        self.statistics_button.setStyleSheet(self.button_style_menu)
        self.statistics_button.clicked.connect(self.toggle_statistics)
        left_layout.addWidget(self.statistics_button)

        self.feature_button = QPushButton("Özellik Oluşturma ve Silme")
        self.feature_button.setStyleSheet(self.button_style_menu)
        self.feature_button.clicked.connect(self.toggle_feature_generation)
        left_layout.addWidget(self.feature_button)

        # Boşluk Doldurma butonu
        self.fill_missing_button = QPushButton("Boşluk Doldurma")
        self.fill_missing_button.setStyleSheet(self.button_style_menu)
        self.fill_missing_button.clicked.connect(self.toggle_fill_missing)
        left_layout.addWidget(self.fill_missing_button)
        # tarih butonu
        self.date_operations_button = QPushButton("Tarih İşlemleri")
        self.date_operations_button.setStyleSheet(self.button_style_menu)
        self.date_operations_button.clicked.connect(self.toggle_date_operations)
        left_layout.addWidget(self.date_operations_button)
        # sınıflandırma butonu
        self.classification_button = QPushButton("Model Eğit")
        self.classification_button.setStyleSheet(self.button_style_menu)
        self.classification_button.clicked.connect(self.toggle_classification)
        left_layout.addWidget(self.classification_button)
        # grafik menüsü butonu
        self.graph_button = QPushButton("Grafik İşlemleri")
        self.graph_button.setStyleSheet(self.button_style_menu)
        self.graph_button.clicked.connect(self.toggle_graph)
        left_layout.addWidget(self.graph_button)
        # feature selection butonu
        self.feature_selection_button = QPushButton("Öznitelik Seçimi")
        self.feature_selection_button.setStyleSheet(self.button_style_menu)
        self.feature_selection_button.clicked.connect(self.toggle_feature_selection)
        left_layout.addWidget(self.feature_selection_button)
        # left_layout.addStretch(1)

        # explain ai butonu
        self.explain_ai_button = QPushButton("Açıklanabilir Yapay Zeka")
        self.explain_ai_button.setStyleSheet(self.button_style_menu)
        self.explain_ai_button.clicked.connect(self.toggle_explain_ai)
        left_layout.addWidget(self.explain_ai_button)

        # Yardım butonu ekleme
        self.help_button = QPushButton("Yardım")
        self.help_button.setStyleSheet(self.button_style_menu)
        self.help_button.clicked.connect(self.show_help_dialog)
        left_layout.addWidget(self.help_button)
        left_layout.addStretch(1)
        self.is_help_dialog_open = False

        main_layout.addLayout(left_layout)

        # İstatistik seçimleri için widget
        self.statistics_widget = QWidget()
        self.statistics_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.checkbox_layout = QVBoxLayout(self.statistics_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # CheckBox'ları yukarı hizala
        self.statistics_widget.setVisible(False)
        options = [
            "Varyans",
            "Kovaryans",
            "Genel Dağılım Ölçüleri",
            "Korelasyon",
            "Histogram ve Normal Dağılım Grafiği"
        ]
        self.statistics_checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            self.checkbox_layout.addWidget(checkbox)
            self.statistics_checkboxes[option] = checkbox

        main_layout.addWidget(self.statistics_widget)

        # Feature Generation için widget
        self.feature_generation_widget = QWidget()
        self.feature_generation_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.feature_layout = QVBoxLayout(self.feature_generation_widget)
        self.feature_layout.setAlignment(Qt.AlignTop)
        self.feature_generation_widget.setVisible(False)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_layout.addWidget(self.feature_list)
        self.button_feat = """
            QPushButton {
                background-color: #1abc9c;
                color: white;
                border-style: solid;
                border-width: 2px;
                border-radius: 10px;
                border-color: #16a085;
                padding: 6px;
                font-size: 16px;
                text-align: center;
                font-weight: bold;
                margin: 4px 2px;
                transition: background-color 0.3s, border-color 0.3s;
            }
            QPushButton:hover {
                background-color: #13b59b;
                border-color: #13b59b;
            }
            QPushButton:pressed {
                background-color: #128c7e;
                border-color: #128c7e;
            }
        """
        self.show_features_button = QPushButton("Özellikleri Gör")
        self.show_features_button.setStyleSheet(self.button_feat)
        self.show_features_button.clicked.connect(self.show_features)
        self.feature_layout.addWidget(self.show_features_button)

        self.feature_combobox = QComboBox()
        self.feature_combobox.addItems(["Toplama", "Çarpma", "Fark", "Bölme", "Concat"])
        self.feature_layout.addWidget(self.feature_combobox)

        self.feature_name_input = QLineEdit()
        self.feature_name_input.setPlaceholderText("Yeni Özellik Adı")
        self.feature_layout.addWidget(self.feature_name_input)

        self.generate_feature_button = QPushButton("Yeni Özellik Oluştur")
        self.generate_feature_button.setStyleSheet(self.button_feat)
        self.generate_feature_button.clicked.connect(self.generate_feature)
        self.feature_layout.addWidget(self.generate_feature_button)

        self.delete_feature_button = QPushButton("Özellik Sil")
        self.delete_feature_button.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border-style: solid;
                border-width: 2px;
                border-radius: 10px;
                border-color: #a93226;
                padding: 6px;
                font-size: 16px;
                text-align: center;
                font-weight: bold;
                margin: 4px 2px;
                transition: background-color 0.3s, border-color 0.3s;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                border-color: #cd6155;
            }
            QPushButton:pressed {
                background-color: #b03a2e;
                border-color: #922b21;
            }
        """)
        self.delete_feature_button.clicked.connect(self.delete_feature)
        self.feature_layout.addWidget(self.delete_feature_button)

        main_layout.addWidget(self.feature_generation_widget)
        # fill_missing için widget
        self.fill_missing_widget = QWidget()
        self.fill_missing_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.checkbox_layout = QVBoxLayout(self.fill_missing_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # CheckBox'ları yukarı hizala
        self.fill_missing_widget.setVisible(False)
        options = [
            "0 ile doldurma",
            "Mean ile doldurma",
            "Mod ile doldurma",
            "FFILL ile doldurma",
            "BFILL ile doldurma",
            "Lineer Interpolasyon ile doldurma",
            "KNN ile doldurma"
        ]
        self.fill_checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            self.checkbox_layout.addWidget(checkbox)
            self.fill_checkboxes[option] = checkbox
        self.missing_columns_combobox = QComboBox()
        self.fill_missing_widget.layout().insertWidget(0, self.missing_columns_combobox)

        main_layout.addWidget(self.fill_missing_widget)
        # sınıflandırma widget
        self.classification_widget = QWidget()
        self.classification_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.checkbox_layout = QVBoxLayout(self.classification_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # CheckBox'ları yukarı hizala
        self.classification_widget.setVisible(False)
        options = [
            "Random Forest",
            "Random Forest ile feature importance",
            "Xgboost",
            "SVM",
            "KNN",
            "Naive Bayes",
            "CatBoost"
        ]
        self.classification_checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            self.checkbox_layout.addWidget(checkbox)
            self.classification_checkboxes[option] = checkbox

        main_layout.addWidget(self.classification_widget)
        # grafik widget
        self.graph_widget = QWidget()
        self.graph_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.checkbox_layout = QVBoxLayout(self.graph_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # CheckBox'ları yukarı hizala
        self.graph_widget.setVisible(False)
        options = [
            "Random Forest Feature Importance",
            "Korelasyon Matrisi",
            "Aykırı Değer"
        ]
        self.graph_checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            self.checkbox_layout.addWidget(checkbox)
            self.graph_checkboxes[option] = checkbox

        main_layout.addWidget(self.graph_widget)

        self.feature_selection_widget = QWidget()
        self.feature_selection_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.checkbox_layout = QVBoxLayout(self.feature_selection_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # CheckBox'ları yukarı hizala
        self.feature_selection_widget.setVisible(False)
        options = [
            "PCA",
            "CHI Kare",
            "FISHER",
            "F-score",
            "Korelasyon tabanlı seçim"
        ]
        self.feature_selection_checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            self.checkbox_layout.addWidget(checkbox)
            self.feature_selection_checkboxes[option] = checkbox

        main_layout.addWidget(self.feature_selection_widget)
        # explain widget
        self.explain_ai_widget = QWidget()
        self.explain_ai_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.checkbox_layout = QVBoxLayout(self.explain_ai_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # CheckBox'ları yukarı hizala
        self.explain_ai_widget.setVisible(False)

        options = [
            "SHAP",
            "LIME"
        ]
        self.explain_ai_checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            self.checkbox_layout.addWidget(checkbox)
            self.explain_ai_checkboxes[option] = checkbox

        main_layout.addWidget(self.explain_ai_widget)
        # Sağ tarafta çıktılar için bir bölüm
        self.right_layout = QVBoxLayout()

        self.output_text = QTextEdit()
        self.output_text.setStyleSheet("background-color: #808080; color: white;")
        self.right_layout.addWidget(self.output_text, 60)  # Çıktı alanını %60 genişliğinde ayarla

        # Tarih İşlemleri widget'ını başlangıçta gizli olarak oluştur
        self.datetime_analysis_widget = DateTimeAnalysisWidget(self.filename, self.output_text)

        self.date_operations_widget = QWidget()
        self.date_operations_layout = QVBoxLayout(self.date_operations_widget)
        self.date_operations_widget.setLayout(self.date_operations_layout)
        self.date_operations_widget.setStyleSheet("background-color: #9370db; color: white;")
        self.date_operations_widget.setVisible(False)

        # Tarih İşlemleri menüsüne eklenecek bileşenler
        self.date_operations_label = QLabel("Tarih/Timestamp Sütunu:")
        self.date_operations_layout.addWidget(self.date_operations_label)

        self.date_operations_combobox = QComboBox()
        self.date_operations_combobox.addItems(["yıl", "ay", "gün"])
        self.date_operations_layout.addWidget(self.date_operations_combobox)

        # Tarih İşlemleri widget'ının içinde Hesapla butonu oluşturun
        self.calculate_button = QPushButton("SEÇ")
        self.calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #1abc9c;
                        color: white;
                        border-style: solid;
                        border-width: 2px;
                        border-radius: 15px;
                        border-color: #16a085;
                        padding: 10px 24px;
                        font-size: 18px;
                        font-weight: bold;
                        text-align: center;
                        transition: background-color 0.3s, border-color 0.3s, transform 0.1s;
                    }
                    QPushButton:hover {
                        background-color: #13b59b;
                        border-color: #13b59b;
                        transform: scale(1.05);
                    }
                    QPushButton:pressed {
                        background-color: #128c7e;
                        border-color: #128c7e;
                        transform: scale(0.95);
                    }
                """)
        self.date_operations_layout.addWidget(self.calculate_button)

        # Burada calculate_button'a bağlantıyı kurun
        self.calculate_button.clicked.connect(self.sync_and_calculate)
        main_layout.addWidget(self.date_operations_widget)
        main_layout.addLayout(self.right_layout)

    def show_help_dialog(self):
        if not self.is_help_dialog_open:
            self.help_dialog = HelpDialog(self)
            self.help_dialog.show()
            self.is_help_dialog_open = True
            self.help_button.setStyleSheet(self.active_style)  # Aktif stil
        else:
            self.help_dialog.raise_()  # Diyalog zaten açıksa ön plana getir
    def help_dialog_closed(self):
        self.is_help_dialog_open = False
        self.help_button.setStyleSheet(self.button_style_menu)  # Normal stile geri dön
    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Veri Seti Seç", "", "CSV Dosyaları (*.csv)", options=options)
        if file_name:
            self.output_text.append(f"Seçilen dosya: {file_name}")
            self.filename = file_name
            self.datetime_analysis_widget.update_widget(file_name)
            self.load_missing_value_columns()  # Eksik değer içeren sütunları yükleme fonksiyonunu çağır

    def toggle_statistics(self):
        self.statistics_widget.setVisible(not self.statistics_widget.isVisible())

        if self.statistics_widget.isVisible():
            self.statistics_button.setStyleSheet(self.active_style)
        else:
            self.statistics_button.setStyleSheet(self.button_style_menu)

        if self.statistics_widget.isVisible():

            # Eğer istatistik widget'ı görünürse, mevcut göster tuşunu temizle
            for i in reversed(range(self.statistics_widget.layout().count())):
                widget = self.statistics_widget.layout().itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() == "Göster":
                    widget.deleteLater()

            # Yeni göster tuşunu ekle
            self.show_button = QPushButton("Göster")
            self.show_button.setStyleSheet("""
    QPushButton {
        background-color: #1abc9c;
        color: white;
        border-style: solid;
        border-width: 2px;
        border-radius: 20px;
        border-color: #159b8a;
        padding: 8px 20px;
        font-size: 16px;
        font-weight: 500;
        text-align: center;
        transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
    }
    QPushButton:hover {
        background-color: #17c9b2;
        border-color: #13b3a6;
        transform: scale(1.03);
    }
    QPushButton:pressed {
        background-color: #138d7a;
        border-color: #117a6f;
        transform: scale(0.97);
    }
""")
            self.show_button.clicked.connect(self.show_statistics)
            self.statistics_widget.layout().addWidget(self.show_button)

    def toggle_feature_generation(self):
        self.feature_generation_widget.setVisible(not self.feature_generation_widget.isVisible())
        if self.feature_generation_widget.isVisible():
            self.feature_button.setStyleSheet(self.active_style)
        else:
            self.feature_button.setStyleSheet(self.button_style_menu)

        if self.feature_generation_widget.isVisible() and self.filename:
            # İşlem seçildiğinde özellik listesini güncelle
            self.show_features()

    def show_statistics(self):

        if not hasattr(self, 'filename'):
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return

        veri_seti = pd.read_csv(self.filename, low_memory=False)
        selected_options = [option for option, checkbox in self.statistics_checkboxes.items() if checkbox.isChecked()]

        if not selected_options:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir istatistik seçin.")
            return

        # Eski tablo varsa kaldır
        for i in reversed(range(self.right_layout.count())):
            widget = self.right_layout.itemAt(i).widget()
            if isinstance(widget, QTableWidget):
                self.right_layout.removeWidget(widget)
                widget.deleteLater()

        table = QTableWidget()
        table.setStyleSheet("color: black;")
        table.setStyleSheet("""
        QTableWidget {
            border: 2px solid black;
        }

        QHeaderView::section {
            background-color: #64B5F6;  /* Başlık için mavi bir arka plan rengi */
            padding: 4px;
            border: 1px solid black;
            font-size: 14px;
            font-weight: bold;
        }

        QTableWidget::item {
            border: 1px solid black;    /* Hücreler için kenarlık */
        }

        QTableWidget::item:selected {
            background-color: #5DADE2;  /* Seçili öğe için arka plan rengi */
        }
        """)

        for option in selected_options:
            if option == "Varyans":
                data_to_display = veri_seti.var()
                data_to_display = pd.DataFrame(data_to_display, columns=['Varyans'])
            elif option == "Kovaryans":
                data_to_display = veri_seti.cov()
            elif option == "Genel Dağılım Ölçüleri":
                data_to_display = veri_seti.describe()
            elif option == "Korelasyon":
                data_to_display = veri_seti.corr()
            elif option == "Histogram ve Normal Dağılım Grafiği":
                column_names = veri_seti.columns.tolist()

                # Dialog oluştur
                dialog = QDialog(self)
                dialog.setWindowTitle("Sütun Seçimi")
                layout = QVBoxLayout(dialog)

                combobox = QComboBox()
                combobox.addItems(column_names)
                layout.addWidget(combobox)

                select_button = QPushButton("Seç")
                layout.addWidget(select_button)

                # Seç butonu tıklandığında çalışacak fonksiyon
                def on_select_clicked():
                    column_name = combobox.currentText()
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.histplot(veri_seti[column_name], kde=True, bins=30, color='skyblue', ax=ax)
                    ax.set_title(f'Histogram and Normal Dağılım Grafiği {column_name}')

                    xmin, xmax = veri_seti[column_name].min(), veri_seti[column_name].max()
                    x = np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, veri_seti[column_name].mean(), veri_seti[column_name].std())
                    ax.plot(x, p, 'k', linewidth=2)

                    dialog.close()

                    histogram_dialog = HistogramDialog(fig, self)
                    histogram_dialog.exec_()

                    # Histogram grafiğini kaydetme seçeneği ekleyelim
                    save_button = QPushButton("Histogram Grafiğini Kaydet")
                    save_button.setStyleSheet("""
                        QPushButton {
                            background-color: #1abc9c;
                            color: white;
                            border-style: solid;
                            border-width: 2px;
                            border-radius: 25px;
                            border-color: #17a589;
                            padding: 10px 15px;
                            font-size: 17px;
                            font-weight: bold;
                            text-align: center;
                            transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
                        }
                        QPushButton:hover {
                            background-color: #48c9b0;
                            border-color: #1abc9c;
                            transform: scale(1.05);
                        }
                        QPushButton:pressed {
                            background-color: #16a085;
                            border-color: #148f77;
                            transform: scale(0.95);
                        }
                    """)

                    save_button.clicked.connect(lambda: self.save_histogram_figure(fig, column_name))
                    self.statistics_widget.layout().addWidget(save_button)

                select_button.clicked.connect(on_select_clicked)
                dialog.exec_()

        if isinstance(data_to_display, pd.DataFrame):  # DataFrame kontrolü yapılıyor.
            self.dataframe_to_table(table, data_to_display)
            self.right_layout.insertWidget(0, table)
            close_button = QPushButton("Tabloyu Kapat")
            close_button.clicked.connect(lambda: self.close_table(table, close_button))
            self.right_layout.addWidget(close_button)
            self.right_layout.addWidget(self.output_text)

    def dataframe_to_table(self, table, dataframe):
        table.setRowCount(dataframe.shape[0])
        table.setColumnCount(dataframe.shape[1])

        # Sütun başlıklarını ayarlayın
        table.setHorizontalHeaderLabels(dataframe.columns.tolist())

        # Indeks başlıklarını ayarlayın (describe() fonksiyonu için ölçümlerin adları burada yer alacak)
        table.setVerticalHeaderLabels(dataframe.index.tolist())

        for i in range(dataframe.shape[0]):
            for j in range(dataframe.shape[1]):
                # .iloc kullanarak değerleri alın ve QTableWidgetItem olarak ekleyin
                table.setItem(i, j, QTableWidgetItem(str(dataframe.iloc[i, j])))

        table.resizeColumnsToContents()

    def close_table(self, table, button):
        # Tablo widget'ını ve kapat butonunu kaldır
        self.right_layout.removeWidget(table)
        self.right_layout.removeWidget(button)

        # Widget'ları sil
        table.deleteLater()
        button.deleteLater()

    def save_histogram_figure(self, figure, column_name):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Histogram Grafiğini Kaydet", "",
                                                   "Resim Dosyaları (*.png *.jpg *.jpeg *.pdf)", options=options)
        if file_name:
            figure.savefig(file_name)
            self.output_text.append(f"Histogram grafiği başarıyla kaydedildi: {file_name}")

    def show_features(self):
        if not hasattr(self, 'filename') or not self.filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return

        veri_seti = pd.read_csv(self.filename, low_memory=False)

        # Hangi işlemin seçildiğini kontrol et
        operation = self.feature_combobox.currentText()

        # Sayısal sütunları veya tüm sütunları göstermek için koşullu mantık
        if operation == "Concat":
            # Concat için tüm sütunları listeye ekleyin
            columns = veri_seti.columns.tolist()
        else:
            # Diğer işlemler için sadece sayısal sütunları listeye ekleyin
            numeric_columns = veri_seti.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_columns

        self.feature_list.clear()
        self.feature_list.addItems(columns)

    def generate_feature(self):
        if not hasattr(self, 'filename'):
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return
        # self.feature_name_input.setEnabled(True)
        # self.feature_name_input.setFocus()
        veri_seti = pd.read_csv(self.filename, low_memory=False)
        selected_columns = [item.text() for item in self.feature_list.selectedItems()]
        operation = self.feature_combobox.currentText()
        new_feature_name = self.feature_name_input.text()

        if len(selected_columns) != 2:
            QMessageBox.warning(self, "Uyarı", "Lütfen iki sütun seçin.")
            return

        if not new_feature_name:
            QMessageBox.warning(self, "Uyarı", "Lütfen yeni özellik için bir isim girin.")
            return

        if operation == "Toplama":
            new_feature = veri_seti[selected_columns[0]] + veri_seti[selected_columns[1]]
        elif operation == "Çarpma":
            new_feature = veri_seti[selected_columns[0]] * veri_seti[selected_columns[1]]
        elif operation == "Fark":
            new_feature = veri_seti[selected_columns[0]] - veri_seti[selected_columns[1]]
        elif operation == "Bölme":
            new_feature = veri_seti[selected_columns[0]] / veri_seti[selected_columns[1]]
        elif operation == "Concat":
            new_feature = pd.concat([veri_seti[selected_columns[0]], veri_seti[selected_columns[1]]], axis=1)

        veri_seti[new_feature_name] = new_feature
        self.output_text.append(f"Yeni özellik oluşturuldu: {new_feature_name}")

        save_option = QMessageBox.question(self, "Kaydet", "Yeni veri setini kaydetmek ister misiniz?",
                                           QMessageBox.Yes | QMessageBox.No)
        if save_option == QMessageBox.Yes:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Yeni Veri Seti Kaydet", "", "CSV Dosyaları (*.csv)",
                                                       options=options)
            if file_name:
                veri_seti.to_csv(file_name, index=False)
                self.output_text.append(f"Yeni veri seti başarıyla kaydedildi: {file_name}")

    def delete_feature(self):
        if not hasattr(self, 'filename'):
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return

        veri_seti = pd.read_csv(self.filename, low_memory=False)
        selected_features = [item.text() for item in self.feature_list.selectedItems()]

        if not selected_features:
            QMessageBox.warning(self, "Uyarı", "Lütfen silmek için bir özellik seçin.")
            return

        veri_seti.drop(selected_features, axis=1, inplace=True)

        save_option = QMessageBox.question(self, "Kaydet", "Değişiklikleri kaydetmek ister misiniz?",
                                           QMessageBox.Yes | QMessageBox.No)
        if save_option == QMessageBox.Yes:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Veri Seti Kaydet", "", "CSV Dosyaları (*.csv)",
                                                       options=options)
            if file_name:
                veri_seti.to_csv(file_name, index=False)
                self.output_text.append(f"Değişiklikler başarıyla kaydedildi: {file_name}")

    def clean_data(self):
        if not hasattr(self, 'filename'):
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return

        son_hali = pd.read_csv(self.filename)

        # Sonsuz değerlere sahip satırları kaldırın
        son_hali = son_hali.dropna()

        # Yeniden indeksleyin (isteğe bağlı)
        son_hali.reset_index(drop=True, inplace=True)

        # Sonuçları başka bir dosyaya kaydetme (isteğe bağlı)
        son_hali.to_csv('temizlenmis_veri.csv', index=False)

        sonsuz_deger_sayisi = son_hali.isnull().sum().sum()

        print("Veri çerçevenizdeki toplam sonsuz değer sayısı:", sonsuz_deger_sayisi)

        # Veri çerçevenizin sütunlarının benzersiz değerlerini sayın
        unique_value_counts = son_hali.nunique()

        # Hangi sütunlarda sadece bir benzersiz değere sahip olduğunu belirleyin
        single_valued_columns = unique_value_counts[unique_value_counts == 1].index

        # Yalnızca tek benzersiz değere sahip olan sütunları kaldırın
        son_hali = son_hali.drop(columns=single_valued_columns)
        son_hali.columns = son_hali.columns.str.strip()
        son_hali = son_hali.replace([np.inf, -np.inf], np.nan)
        son_hali.info()

        # z skoru yaklaşım aykırı değer
        from scipy import stats

        # Sadece sayısal sütunları seçin
        sayisal_sutunlar = son_hali.select_dtypes(include=[np.number])

        # Aykırı değer eşik değeri (örneğin, 3) belirleyin
        aykiri_esik = 3

        # Her sütun için aykırı değerleri bulun ve yazdırın
        for sutun in sayisal_sutunlar.columns:
            z_scores = np.abs(stats.zscore(sayisal_sutunlar[sutun]))
            aykiri_satirlar = np.where(z_scores > aykiri_esik)
            # son_hali = son_hali.drop(son_hali.index[aykiri_satirlar[0]]) verisetinden aykırı değerleri kaldırır

    def toggle_fill_missing(self):
        self.fill_missing_widget.setVisible(not self.fill_missing_widget.isVisible())

        if self.fill_missing_widget.isVisible():
            self.fill_missing_button.setStyleSheet(self.active_style)
        else:
            self.fill_missing_button.setStyleSheet(self.button_style_menu)


        if self.fill_missing_widget.isVisible():

            for i in reversed(range(self.fill_missing_widget.layout().count())):
                widget = self.fill_missing_widget.layout().itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() == "Doldur":
                    widget.deleteLater()

            # Yeni göster tuşunu ekle
            self.show_button = QPushButton("Doldur")
            self.show_button.setStyleSheet("""
                QPushButton {
                    background-color: #1abc9c;
                    color: white;
                    border-style: solid;
                    border-width: 2px;
                    border-radius: 20px;
                    border-color: #159b8a;
                    padding: 8px 20px;
                    font-size: 16px;
                    font-weight: 500;
                    text-align: center;
                    transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
                }
                QPushButton:hover {
                    background-color: #17c9b2;
                    border-color: #13b3a6;
                    transform: scale(1.03);
                }
                QPushButton:pressed {
                    background-color: #138d7a;
                    border-color: #117a6f;
                    transform: scale(0.97);
                }
            """)
            self.show_button.clicked.connect(self.fill_missing)
            self.fill_missing_widget.layout().addWidget(self.show_button)

    def fill_missing(self):
        if not hasattr(self, 'filename') or not self.filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return
        self.output_text.clear()
        df = pd.read_csv(self.filename)

        df_copy1=df.copy()
        # Orijinal dosyayı yeni bir isimle kaydet


        # Eksik veri bilgisini hesapla
        missing_info = df_copy1.isnull().sum()
        missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
        self.output_text.setStyleSheet("font-size: 14pt; font-family: Arial; color:black")
        # Eksik veri bilgisini arayüzdeki metin alanına yazdır
        if missing_info.empty:
            self.output_text.append("<b>Eksik veri yok.</b>")
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Dolu Veriyi Kaydet", "",
                                                       "CSV Dosyaları (*.csv)", options=options)
            if file_name:  # file_name tanımlı ve boş değilse
                df_copy1.to_csv(file_name, index=False)
                self.cleaned_data_filename = file_name  # Yeni oluşturulan dosyanın adını saklayın
                QMessageBox.information(self, "Başarılı", "Dosya başarıyla kaydedildi.")
            else:
                self.output_text.append("Kaydetme işlemi iptal edildi veya dosya adı girilmedi.")


        else:
            missing_data_text = "<b>Eksik Veri Bilgisi:</b><br>"
            for column, missing_count in missing_info.items():
                missing_data_text += f"• {column}: <span style='color:red'>{missing_count}</span> adet eksik<br>"

            self.output_text.append(missing_data_text)


        selected_method = None
        selected_column = self.missing_columns_combobox.currentText()
        if not selected_column:
            QMessageBox.warning(self, "Uyarı", "Lütfen eksik değer içeren bir sütun seçin.")
            return
        for option, checkbox in self.fill_checkboxes.items():
            if checkbox.isChecked():
                selected_method = option
                break

        if selected_method is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir doldurma yöntemi seçin.")
            return
        filled_rows = []
        #df = pd.read_csv(self.filename)
        try:
            if selected_method == "Mean ile doldurma" and not pd.api.types.is_numeric_dtype(df_copy1[selected_column]):
                raise ValueError("Mean ile doldurma yalnızca sayısal sütunlar için uygundur.")
            if selected_method == "Mod ile doldurma" and pd.api.types.is_numeric_dtype(df_copy1[selected_column]):
                raise ValueError("Mod ile doldurma yalnızca kategorik veya metinsel sütunlar için uygundur.")
            if selected_method == "FFILL ile doldurma" and pd.api.types.is_datetime64_any_dtype(
                    df_copy1[selected_column]):
                raise ValueError("FFILL ile doldurma tarih ve zaman sütunları için uygun değildir.")
            if selected_method == "BFILL ile doldurma" and pd.api.types.is_datetime64_any_dtype(
                    df_copy1[selected_column]):
                raise ValueError("BFILL ile doldurma tarih ve zaman sütunları için uygun değildir.")
            if selected_method == "0 ile doldurma" and pd.api.types.is_string_dtype(df_copy1[selected_column]):
                raise ValueError("0 ile doldurma metinsel sütunlar için uygun değildir.")
            if selected_method == "Lineer Interpolasyon ile doldurma" and not pd.api.types.is_numeric_dtype(
                    df_copy1[selected_column]):
                raise ValueError("Lineer Interpolasyon ile doldurma yalnızca sayısal sütunlar için uygundur.")
            if selected_method == "KNN ile doldurma" and not (
                    pd.api.types.is_numeric_dtype(df_copy1[selected_column]) or pd.api.types.is_categorical_dtype(
                    df_copy1[selected_column])):
                raise ValueError("KNN ile doldurma yalnızca sayısal veya kategorik sütunlar için uygundur.")
        # 0 ile doldurma

            if selected_method=="0 ile doldurma":
                missing_before = df_copy1[selected_column].isnull()
                df_copy1[selected_column].fillna(0, inplace=True)
                missing_after = df_copy1[selected_column].isnull()
                filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            # ortalama ile doldurma
            if selected_method=="Mean ile doldurma":
                if df_copy1[selected_column].dtype == np.number:
                    missing_before = df_copy1[selected_column].isnull()
                    df_copy1[selected_column].fillna(df_copy1[selected_column].mean(), inplace=True)
                    missing_after = df_copy1[selected_column].isnull()
                    filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            # Mod ile doldurma
            if selected_method=="Mod ile doldurma":
                if df_copy1[selected_column].dtype == object or df_copy1[selected_column].dtype == 'category':
                    missing_before = df_copy1[selected_column].isnull()
                    df_copy1[selected_column].fillna(df_copy1[selected_column].mode()[0], inplace=True)
                    missing_after = df_copy1[selected_column].isnull()
                    filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            # ffill ile doldurma
            if selected_method=="FFILL ile doldurma":
                missing_before = df_copy1[selected_column].isnull()
                df_copy1[selected_column].fillna(method='ffill', inplace=True)
                missing_after = df_copy1[selected_column].isnull()
                filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            # bfill ile doldurma
            if selected_method=="BFILL ile doldurma":
                missing_before = df_copy1[selected_column].isnull()
                df_copy1[selected_column].fillna(method='bfill', inplace=True)
                missing_after = df_copy1[selected_column].isnull()
                filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            # Lineer Interpolasyon ile doldurma
            if selected_method=="Lineer Interpolasyon ile doldurma":
                missing_before = df_copy1[selected_column].isnull()
                df_copy1[selected_column].interpolate(method='linear', inplace=True)
                missing_after = df_copy1[selected_column].isnull()
                filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            # KNN ile doldurma
            if selected_method=="KNN ile doldurma":
                missing_before = df_copy1[selected_column].isnull()

                selected_data = df_copy1[[selected_column]]

                # Sonsuz değerleri NaN ile değiştir
                selected_data.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Eğer seçilen sütun sayısal bir sütunsa, KNN doldurma işlemini uygula
                if selected_data[selected_column].dtype == np.number:
                    # KNN Imputer'ı yalnızca seçilen sütun için fit ve transform yap
                    imputer = KNNImputer(n_neighbors=5)
                    transformed_data = imputer.fit_transform(selected_data)

                    # Doldurulan verileri orijinal DataFrame'e geri yerleştir
                    df_copy1[selected_column] = transformed_data.ravel()
                    missing_after = df[selected_column].isnull()
                    filled_rows = (missing_before & ~missing_after).index[missing_before & ~missing_after].tolist()

            filled_rows_text = ", ".join(map(str, filled_rows))
            self.output_text.append(
                f"{selected_column} sütunu, {selected_method} yöntemi ile dolduruldu. Doldurulan satırlar: {filled_rows_text}\n")

            self.display_missing_value_info(df)
            df_copy1.to_csv(self.filename, index=False)

            self.load_missing_value_columns()  # Eksik değer içeren sütun listesini yeniden yükle

            # İşlem sonucunu göster
            QMessageBox.information(self, "Başarılı", "Eksik değerler dolduruldu.")
            if not self.missing_columns_combobox.count():
                options = QFileDialog.Options()
                file_name, _ = QFileDialog.getSaveFileName(self, "Düzenlenmiş Veriyi Kaydet", "",
                                                           "CSV Dosyaları (*.csv)", options=options)
            if file_name:
                df_copy1.to_csv(file_name, index=False)
                self.cleaned_data_filename = file_name  # Yeni oluşturulan dosyanın adını saklayın
                QMessageBox.information(self, "Başarılı", "Dosya başarıyla kaydedildi.")
        except ValueError as e:
            QMessageBox.warning(self, "Uyarı", str(e))

    def display_missing_value_info(self, df):
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
        # self.output_text.append("Eksik veri bilgisi:\n")
        for column, missing_count in missing_info.items():
            # self.output_text.append(f"{column}: {missing_count} adet eksik")
            pass

    def load_missing_value_columns(self):
        df = pd.read_csv(self.filename)
        missing_value_columns = df.columns[df.isnull().any()].tolist()
        self.missing_columns_combobox.clear()
        if missing_value_columns:
            self.missing_columns_combobox.addItems(missing_value_columns)
        else:
            QMessageBox.information(self, "Bilgi", "Tüm eksik veriler dolduruldu.")

    def update_feature_list(self):
        if not self.filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dosya seçin.")
            return
        veri_seti = pd.read_csv(self.filename, low_memory=False)
        self.feature_list.clear()

        selected_column_type = self.column_type_combobox.currentText()
        if selected_column_type == "Sayısal Sütunlar":
            numeric_columns = veri_seti.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_list.addItems(numeric_columns)
        elif selected_column_type == "Kategorik Sütunlar":
            categorical_columns = veri_seti.select_dtypes(include=['object', 'category']).columns.tolist()
            self.feature_list.addItems(categorical_columns)

    def toggle_date_operations(self):
        self.date_operations_widget.setVisible(not self.date_operations_widget.isVisible())
        if self.date_operations_widget.isVisible():
            self.date_operations_button.setStyleSheet(self.active_style)
        else:
            self.date_operations_button.setStyleSheet(self.button_style_menu)

        # Tarih İşlemleri widget görünür olduğunda Hesapla butonunun bağlantısını kontrol edin
        if self.date_operations_widget.isVisible():
            if not hasattr(self, 'calculate_button_connected'):
                self.calculate_button.clicked.connect(self.datetime_analysis_widget.calculate_data_count)
                self.calculate_button_connected = True

    def sync_and_calculate(self):
        # date_operations_combobox'un seçili değerini al
        selected_unit = self.date_operations_combobox.currentText()
        # DateTimeAnalysisWidget içindeki unit_combobox'a bu değeri ayarla
        self.datetime_analysis_widget.unit_combobox.setCurrentText(selected_unit)
        # Sonra hesaplamayı yap
        self.datetime_analysis_widget.add_datetime_feature()

    def toggle_classification(self):
        self.classification_widget.setVisible(not self.classification_widget.isVisible())
        if self.classification_widget.isVisible():
            self.classification_button.setStyleSheet(self.active_style)
        else:
            self.classification_button.setStyleSheet(self.button_style_menu)

        if self.classification_widget.isVisible():
            # Burada, eğer daha önce "Göster" butonu eklenmemişse, ekleyin
            if not hasattr(self, 'classification_show_button'):
                self.classification_show_button = QPushButton("Göster")
                self.classification_show_button.setStyleSheet("""
                    QPushButton {
                        background-color: #1abc9c;
                        color: white;
                        border-style: solid;
                        border-width: 2px;
                        border-radius: 20px;
                        border-color: #159b8a;
                        padding: 8px 20px;
                        font-size: 16px;
                        font-weight: 500;
                        text-align: center;
                        transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
                    }
                    QPushButton:hover {
                        background-color: #17c9b2;
                        border-color: #13b3a6;
                        transform: scale(1.03);
                    }
                    QPushButton:pressed {
                        background-color: #138d7a;
                        border-color: #117a6f;
                        transform: scale(0.97);
                    }
                """)
                self.classification_show_button.clicked.connect(self.classification)
                self.classification_widget.layout().addWidget(self.classification_show_button)
                setattr(self, 'classification_show_button_added', True)

    def classification(self):
        selected_method = None
        for checkbox_text, checkbox in self.classification_checkboxes.items():
            if checkbox.isChecked():
                selected_method = checkbox_text
                break

        if selected_method == "Random Forest":
            self.apply_random_forest()
        elif selected_method == "Random Forest ile feature importance":
            self.apply_random_forest_feature_importance()
        elif selected_method == "Korelasyon Sıralaması":
            self.apply_correlation_ranking()
        elif selected_method == "Xgboost":
            self.apply_xgboost()
        elif selected_method=="SVM":
            self.apply_train_svm()
        elif selected_method=="KNN":
            self.apply_knn()
        elif selected_method=="Naive Bayes":
            self.apply_naive_bayes()
        elif selected_method=="CatBoost":
            self.apply_catboost()
        else:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir sınıflandırma yöntemi seçin.")

    def apply_catboost(self):
        dialog = CatBoostParameterDialog(self)
        if dialog.exec():
            params = dialog.get_parameters()

        self.output_text.clear()
        data = pd.read_csv(self.cleaned_data_filename)

        unique_value_counts = data.nunique()
        single_valued_columns = unique_value_counts[unique_value_counts == 1].index
        data = data.drop(columns=single_valued_columns)
        data.columns = data.columns.str.strip()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)

        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 data.columns[data.dtypes == 'object'].tolist(), 0, False)
        if not ok or not target_column:
            return

        if data[target_column].dtype != 'object':
            QMessageBox.warning(self, "Hata", "Seçilen hedef değişken kategorik bir değişken olmalıdır.")
            return

        cat_features = [i for i, col in enumerate(data.columns) if data[col].dtype == 'object']
        target_index = data.columns.get_loc(target_column)
        if target_index in cat_features:
            cat_features.remove(target_index)  # Hedef sütunu kategorik özellikler listesinden çıkar

        X = data.drop(columns=[target_column])
        y = data[target_column].astype('category').cat.codes
        classes = data[target_column].unique()
        y_bin = label_binarize(y, classes=classes)
        n_classes = y_bin.shape[1]

        # Veriyi eğitim, doğrulama ve test setlerine ayır
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                                    random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=params['val_size'],
                                                          random_state=42)

        model = CatBoostClassifier(
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            iterations=params['iterations'],
            early_stopping_rounds=params['early_stopping_rounds'],
            l2_leaf_reg=params['l2_leaf_reg'],
            border_count=params['border_count'],
            bagging_temperature=params['bagging_temperature'],
            cat_features=cat_features,
            eval_metric='Accuracy',
            verbose=True
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        # Doğrulama seti üzerinde performansı değerlendir
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        self.output_text.append(f"Validation Setinin Doğruluk Oranı: {val_accuracy}")
        self.output_text.append(
            f"Validation Seti Sınıflandırma Raporu:\n{classification_report(y_val, y_val_pred)}")
        self.output_text.append(f"Validation Seti Confusion Matrix:\n{confusion_matrix(y_val, y_val_pred)}")

        # Test seti üzerinde performansı değerlendir
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.output_text.append(f"Test Setinin Doğruluk Oranı: {accuracy}")
        self.output_text.append(
            f" Test Seti Sınıflandırma Raporu:\n{classification_report(y_test, y_pred)}")
        self.output_text.append(f"Test Seti Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        num_classes = np.unique(y_train).size

        # y_test için manuel binarize işlemi
        y_test_binarized = np.zeros((y_test.size, num_classes))
        for i, unique_value in enumerate(np.unique(y_train)):
            y_test_binarized[:, i] = (y_test == unique_value).astype(int)
        y_pred_proba = model.predict_proba(X_test)

        # Şimdi ROC hesaplaması yapabilirsiniz
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        dialog = QDialog()
        dialog.setWindowTitle("ROC CatBoost")

        # Figure ve canvas oluştur
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # ROC Eğrisi çiz
        colors = cycle(
            ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen', 'gray',
             'cyan'])
        for i, color in zip(range(len(classes)), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Roc CatBoost')
        ax.legend(loc="lower right")

        layout = QVBoxLayout()
        layout.addWidget(canvas)

        # Kaydetme butonu
        save_button = QPushButton('Grafiği Kaydet')
        save_button.clicked.connect(lambda: fig.savefig(
            QFileDialog.getSaveFileName(dialog, 'Grafiği Kaydet', filter='PNG Files (*.png);;JPG Files (*.jpg)')[0]))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

        # Eğitim sonrası kayıpları çek
        evals_result = model.get_evals_result()

        # Loss grafiği çizimi için ek dialog
        loss_dialog = QDialog()
        loss_dialog.setWindowTitle("Loss Grafikleri")

        fig_loss = Figure(figsize=(10, 8))
        canvas_loss = FigureCanvas(fig_loss)
        ax_loss = fig_loss.add_subplot(111)

        # Eğitim ve doğrulama loss değerlerini çiz
        ax_loss.plot(evals_result['learn']['MultiClass'], label='Train Loss')
        ax_loss.plot(evals_result['validation']['MultiClass'], label='Validation Loss')
        ax_loss.set_title('Loss Grafikleri')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('MultiClass Loss')
        ax_loss.legend(loc="upper right")

        layout_loss = QVBoxLayout()
        layout_loss.addWidget(canvas_loss)

        save_button_loss = QPushButton('Grafiği Kaydet')
        save_button_loss.clicked.connect(lambda: fig_loss.savefig(
            QFileDialog.getSaveFileName(loss_dialog, 'Grafiği Kaydet', filter='PNG Files (*.png);;JPG Files (*.jpg)')[
                0]))
        layout_loss.addWidget(save_button_loss)

        loss_dialog.setLayout(layout_loss)
        loss_dialog.exec_()

    def apply_naive_bayes(self):
        self.output_text.clear()
        dialog = NaiveBayesParameterDialog(self)
        if dialog.exec():

            params = dialog.get_parameters()
        # Parametre diyalogundan parametreleri al

        # Veri setini yükle
        data_sample = pd.read_csv(self.cleaned_data_filename).sample(frac=1.0, random_state=42)

        # Tek değerli sütunları kaldır
        unique_value_counts = data_sample.nunique()
        single_valued_columns = unique_value_counts[unique_value_counts == 1].index
        data_sample = data_sample.drop(columns=single_valued_columns)
        data_sample.columns = data_sample.columns.str.strip()
        data_sample = data_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Hedef değişkeni seç
        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 data_sample.columns[data_sample.dtypes == 'object'].tolist(), 0, False)
        if not ok or not target_column:
            return
        # Label encoding uygula
        target_names = data_sample[target_column].unique().astype(str).tolist()  # Sınıf isimlerini string olarak al
        # Önce hedef sütunu doğrudan alalım.
        y = data_sample[target_column].copy()

        # Hedef değişkeni doğrudan binarize edelim.
        # Burada y'nin orijinal değerlerini classes argümanına geçiriyoruz.
        classes = data_sample[target_column].unique()
        y_bin = label_binarize(y, classes=classes)
        n_classes = y_bin.shape[1]

        # Diğer işlemlere devam edin
        error_columns = []
        for col in data_sample.select_dtypes(include=['object']).columns.difference([target_column]):
            try:
                data_sample[col] = data_sample[col].astype('category').cat.codes
            except Exception as e:
                # Hata veren sütunu kaydet
                error_columns.append(col)
                data_sample = data_sample.drop(columns=error_columns)

        X = data_sample.drop(columns=[target_column])  # Hedef değişkeni binarize et
        n_classes = y_bin.shape[1]
        for col in data_sample.columns.difference([target_column]):
            # Sütun tipini tarih/zaman tipine dönüştürmeyi dene
            try:
                data_sample[col] = pd.to_datetime(data_sample[col])
                data_sample.drop(col, axis=1, inplace=True)  # Başarılı dönüşüm sonrası sütunu çıkar
            except (ValueError, TypeError):
                continue
        # Veriyi eğitim, doğrulama ve test setlerine ayır
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                                    random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=params['validation_size'],
                                                          random_state=42)  # %10 doğrulama seti

        # Özellik ölçeklendirme
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Modelleri kur ve eğit
        model1 = GaussianNB(var_smoothing=params['alpha'])

        model3 = LogisticRegressionCV(cv=3, max_iter=300, random_state=42, penalty='l2')

        # Voting Classifier oluştur
        voting_clf = VotingClassifier(
            estimators=[('nb', model1),  ('lr', model3)],
            voting='soft')  # 'soft' voting kullanarak olasılıkları temel al
        voting_clf.fit(X_train, y_train)

        # Test seti üzerinde tahmin yap ve performansı değerlendir
        y_pred = voting_clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        confusion = confusion_matrix(y_test, y_pred)
        # Doğrulama seti üzerinde tahmin yap ve performansı değerlendir
        y_val_pred = voting_clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_report = classification_report(y_val, y_val_pred, target_names=target_names)
        val_confusion = confusion_matrix(y_val, y_val_pred)
        # Sonuçları yazdır
        self.output_text.append(f"Test Setinin Doğruluk Oranı: {accuracy}\n")
        self.output_text.append(f"Test Setinin Sınıflandırma Raporu:\n{report}")
        self.output_text.append(f"Test Setinin Confusion Matrix:\n{confusion}")

        self.output_text.append(f"Doğrulama Setinin Doğruluk Oranı: {val_accuracy}\n")
        self.output_text.append(f"Doğrulama Setinin Sınıflandırma Raporu:\n{val_report}")
        self.output_text.append(f"Doğrulama Setinin Confusion Matrix:\n{val_confusion}\n")

        y_pred_prob = voting_clf.predict_proba(X_test)

        # Örnek olarak y'nin sınıf sayısını alıyoruz
        num_classes = np.unique(y_train).size

        # y_test için manuel binarize işlemi
        y_test_binarized = np.zeros((y_test.size, num_classes))
        for i, unique_value in enumerate(np.unique(y_train)):
            y_test_binarized[:, i] = (y_test == unique_value).astype(int)

        # Şimdi ROC hesaplaması yapabilirsiniz
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        dialog = QDialog()
        dialog.setWindowTitle("ROC Curve")

        # Figure ve canvas oluştur
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # ROC Eğrisi çiz
        colors = cycle(
            ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen', 'gray',
             'cyan'])
        for i, color in zip(range(len(target_names)), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Roc Naive Bayes')
        ax.legend(loc="lower right")

        layout = QVBoxLayout()
        layout.addWidget(canvas)

        # Kaydetme butonu
        save_button = QPushButton('Grafiği Kaydet')
        save_button.clicked.connect(lambda: fig.savefig(
            QFileDialog.getSaveFileName(dialog, 'Grafiği Kaydet', filter='PNG Files (*.png);;JPG Files (*.jpg)')[0]))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

        y_val_pred_prob = voting_clf.predict_proba(X_val)

        # Log-loss hesaplayın
        test_log_loss = log_loss(y_test, y_pred_prob)
        val_log_loss = log_loss(y_val, y_val_pred_prob)

        # Log-loss grafiği için dialog oluşturun
        log_loss_dialog = QDialog()
        log_loss_dialog.setWindowTitle("Log-Loss Grafikleri")

        fig_log_loss = Figure(figsize=(10, 8))
        canvas_log_loss = FigureCanvas(fig_log_loss)
        ax_log_loss = fig_log_loss.add_subplot(111)

        # İki log-loss değerini çizin
        ax_log_loss.bar(['Test Log-Loss', 'Validation Log-Loss'], [test_log_loss, val_log_loss],
                        color=['blue', 'orange'])
        ax_log_loss.set_title('Log-Loss')
        ax_log_loss.set_ylabel('Log-Loss')

        layout_log_loss = QVBoxLayout()
        layout_log_loss.addWidget(canvas_log_loss)

        save_button_log_loss = QPushButton('Grafiği Kaydet')
        save_button_log_loss.clicked.connect(lambda: fig_log_loss.savefig(
            QFileDialog.getSaveFileName(log_loss_dialog, 'Grafiği Kaydet',
                                        filter='PNG Files (*.png);;JPG Files (*.jpg)')[0]))
        layout_log_loss.addWidget(save_button_log_loss)

        log_loss_dialog.setLayout(layout_log_loss)
        log_loss_dialog.exec_()

    def get_random_forest_parameters(self):
        dialog = RandomForestParameterDialog(self)
        if dialog.exec():
            return dialog.get_parameters()
        else:
            return None

    def apply_random_forest(self):
        self.output_text.clear()

        # Kullanıcıdan Random Forest parametrelerini al
        rf_params = self.get_random_forest_parameters()
        if rf_params is None:
            return  # Kullanıcı iptal ettiyse işlemi durdur

        # Kullanıcıdan test seti büyüklüğünü alma
        test_size, ok = QInputDialog.getDouble(self, "Test Size", "Enter test size (e.g., 0.2):", min=0.01, max=0.99,
                                               decimals=2)
        if not ok:
            return  # Kullanıcı iptal ettiyse işlemi durdur

        val_size = 0.1 / (1 - test_size)  # test_size dışında kalan verinin %10'u validasyon için kullanılacak

        data_sample = pd.read_csv(self.cleaned_data_filename).sample(frac=1.0,
                                                                     random_state=42)  # Veri setinin %100'sini örnek al

        unique_value_counts = data_sample.nunique()
        #print(unique_value_counts)
        # Hangi sütunlarda sadece bir benzersiz değere sahip olduğunu belirleyin
        single_valued_columns = unique_value_counts[unique_value_counts == 1].index
        #print(single_valued_columns)
        data_sample=data_sample.drop(columns=single_valued_columns)
        data_sample.columns = data_sample.columns.str.strip()
        data_sample = data_sample.replace([np.inf, -np.inf], np.nan).fillna(0)
        error_columns = []
        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 data_sample.columns[data_sample.dtypes == 'object'].tolist(), 0, False)
        if not ok or not target_column:
            return

        if data_sample[target_column].dtype != 'object':
            QMessageBox.warning(self, "Hata", "Seçilen hedef değişken kategorik bir değişken olmalıdır.")
            return
        # Label encoding for the 'Label' column
        self.le = LabelEncoder()
        data_sample[target_column] = self.le.fit_transform(data_sample[target_column])
        label_mapping = dict(zip(self.le.classes_, range(len(self.le.classes_))))
        self.category_mappings = {}
        for col in data_sample.select_dtypes(include=['object']).columns:
            try:

                 data_sample[col] = data_sample[col].astype('category').cat.codes
            except Exception as e:
                # Hata veren sütunu kaydet
                error_columns.append(col)


                data_sample = data_sample.drop(columns=error_columns)


        X = data_sample.drop(columns=[target_column])
        y = data_sample[target_column]
        #data_sample = data_sample.drop(columns=['Timestamp'])
        for col in data_sample.columns.difference([target_column]):
            # Sütun tipini tarih/zaman tipine dönüştürmeyi dene
            try:
                data_sample[col] = pd.to_datetime(data_sample[col])
                data_sample.drop(col, axis=1, inplace=True)  # Başarılı dönüşüm sonrası sütunu çıkar
            except (ValueError, TypeError):
                continue  # Dönüşüm başarısız olursa, sütunu olduğu gibi bırak



        # Veriyi eğitim+validasyon ve test setlerine ayırma
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.X_test = X_test
        # Eğitim+validasyon setini eğitim ve validasyon setlerine ayırma
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)
        self.X_train = X_train
        # RandomForestClassifier oluştururken kullanıcıdan alınan parametreleri kullan
        self.clf = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            bootstrap=rf_params['bootstrap'],
            random_state=42,
            max_samples=rf_params['max_samples'],
            max_features=rf_params['max_features'],
            oob_score=rf_params['oob_score']
        )

        # Modeli eğit
        self.clf.fit(X_train, y_train)
        self.X_train_columns = X_train.columns.tolist()

        # Validasyon seti üzerinde performans metriklerini hesapla ve göster
        y_pred_val = self.clf.predict(X_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        report_val = classification_report(y_val, y_pred_val)
        confusion_val = confusion_matrix(y_val, y_pred_val)

        # Sonuçları yazdır
        self.output_text.append(f"Validasyon Setinin Doğruluk Oranı: {accuracy_val}\n")
        self.output_text.append(
            f"Validasyon Setinin Sınıflandırma Raporu:\n{classification_report(y_val, y_pred_val, target_names=self.le.classes_)}")
        self.output_text.append(f"Validasyon Seti Confusion Matrix:\n{confusion_val}\n")

        # Test seti üzerinde tahmin yapma ve performans metriklerini hesaplama
        y_pred_test = self.clf.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        report_test = classification_report(y_test, y_pred_test)
        confusion_test = confusion_matrix(y_test, y_pred_test)

        self.output_text.append(f"Test Setinin Doğruluk Oranı: {accuracy_test}\n")
        self.output_text.append(
            f"Test Setinin Sınıflandırma Raporu:\n{classification_report(y_test, y_pred_test, target_names=self.le.classes_)}")
        self.output_text.append(f"Test Setinin Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

        # K-Fold çapraz doğrulama ayarları ve hesaplama
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Veriyi 5 katman olarak ayır
        cross_val_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            bootstrap=rf_params['bootstrap'],
            random_state=42
        )
        scores = cross_val_score(cross_val_model, X, y, cv=kf, scoring='accuracy')  # 'accuracy' metriğini kullan
        ...
        # ROC Eğrisi için hazırlık
        y_test_bin = label_binarize(y_test, classes=[i for i in range(len(self.le.classes_))])
        y_scores = self.clf.predict_proba(X_test)

        # ROC hesaplamaları ve grafik için hazırlık
        fpr, tpr, roc_auc = {}, {}, {}
        for i, label in enumerate(self.le.classes_):
            fpr[label], tpr[label], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
            roc_auc[label] = auc(fpr[label], tpr[label])

        # Grafikleri çiz
        fig = Figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Çapraz doğrulama skorları grafiği
        ax1.plot(range(1, 6), scores, marker='o', linestyle='-', color='b')
        ax1.set_title('Çapraz Doğrulama Doğruluk Skoru')
        ax1.set_xlabel('Katlamalar')
        ax1.set_ylabel('Doğruluk')

        # ROC Eğrisi grafiği
        for label in self.le.classes_:
            ax2.plot(fpr[label], tpr[label], label=f'{label} (AUC = {roc_auc[label]:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Eğrisi')
        ax2.legend(loc='lower right')

        # Grafikleri QDialog ile göster
        dialog = QDialog()
        dialog.setWindowTitle('Random Forest Model Değerlendirme')
        dialog_layout = QVBoxLayout()
        canvas = FigureCanvas(fig)
        dialog_layout.addWidget(canvas)

        # Grafik kaydetme butonu
        save_button = QPushButton('Grafik Kaydet', dialog)
        save_button.clicked.connect(lambda: self.save_figure(fig))
        dialog_layout.addWidget(save_button)

        dialog.setLayout(dialog_layout)
        dialog.exec_()
        # Çapraz doğrulama sonuçlarını yazdır
        self.output_text.append("Her bir katmanın doğruluk skoru:\n" + "\n".join([str(score) for score in scores]))
        self.output_text.append("\nOrtalama doğruluk skoru: " + str(scores.mean()))
    def save_figure(self, fig):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Grafik Kaydet", "",
                                                  "PNG Files (*.png);;All Files (*)", options=options)
        if fileName:
            fig.savefig(fileName)
            QMessageBox.information(self, 'Başarılı', 'Grafik başarıyla kaydedildi.')

    def apply_random_forest_feature_importance(self):
        if self.clf is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce Random Forest modelini uygulayın.")
            return

        # Kullanıcıdan gösterilecek özellik sayısını al
        num_features, ok = QInputDialog.getInt(self, "Özellik Sayısı", "Gösterilecek özellik sayısını girin:", min=1,
                                               max=100, step=1)
        if not ok:
            return  # Kullanıcı iptal ettiyse işlemi durdur

        feature_importances = self.clf.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': self.X_train_columns, 'importance': feature_importances})
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

        # Kullanıcının istediği sayıda en önemli özellikleri al
        top_features = feature_importance_df.head(num_features)
        self.output_text.clear()
        self.output_text.append("En Önemli Özellikler:\n")
        self.output_text.append(top_features.to_string(index=False))

    def apply_correlation_ranking(self):
        try:
            if not self.cleaned_data_filename:
                self.output_text.append("Lütfen önce bir dosya seçin ve boşluk doldurma işlemi yapın.")
                return

            data = pd.read_csv(self.cleaned_data_filename)
            data = data.apply(pd.to_numeric, errors='coerce')

            # Sayısal veriler için korelasyon matrisini hesapla
            correlation_matrix = data.corr()

            # Korelasyon matrisinden üst üçgeni alarak tekrar eden korelasyonları kaldır
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            filtered_corr = correlation_matrix.where(mask)

            # Sıralı korelasyon değerlerini al
            corr_unstacked = filtered_corr.unstack()
            sorted_correlations = corr_unstacked.sort_values(ascending=False).dropna()

            # Sonuçları metin olarak yazdır
            result_text = "Korelasyon Sıralaması:\n"
            for (idx, ((col1, col2), corr)) in enumerate(sorted_correlations.items(), 1):
                # Aynı sütunlar arası korelasyonu hariç tut
                if col1 != col2:
                    result_text += f"{idx}. {col1} - {col2}: {corr}\n"

            self.output_text.setText(result_text)
        except Exception as e:
            self.output_text.append(f"Korelasyon sıralaması yapılırken bir hata oluştu: {e}")

    def apply_xgboost(self):
        dialog = XGBoostParameterDialog(self)
        if dialog.exec():
            params = dialog.get_parameters()

        self.output_text.clear()
        # Veri setini yükle
        data = pd.read_csv(self.cleaned_data_filename)

        unique_value_counts = data.nunique()
        single_valued_columns = unique_value_counts[unique_value_counts == 1].index
        data = data.drop(columns=single_valued_columns)
        data.columns = data.columns.str.strip()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)

        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 data.columns[data.dtypes == 'object'].tolist(), 0, False)
        if not ok or not target_column:
            return

        if data[target_column].dtype != 'object':
            QMessageBox.warning(self, "Hata", "Seçilen hedef değişken kategorik bir değişken olmalıdır.")
            return

        for col in data.select_dtypes(include=['object']).columns.difference([target_column]):

            data[col] = data[col].astype('category')

        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

        X = data.drop([target_column], axis=1)
        y = data[target_column]

        # Veriyi eğitim, doğrulama ve test setlerine ayır
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                                    random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=params['val_size'],
                                                          random_state=42)

        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],  # Örnek: Max derinlik ayarını değiştir
            min_child_weight=params['min_child_weight'],  # Minimum çocuk ağırlığını değiştir
            gamma=params['gamma'],  # Gamma değerini ayarla
            subsample=params['subsample'],  # Altörneklem oranını ayarla
            colsample_bytree=params['colsample_bytree'],  # Ağaç başına sütun örnekleme oranını ayarla
            reg_alpha=params['reg_alpha'],  # L1 düzenlileştirmeyi ayarla
            reg_lambda=params['reg_lambda'],# L2 düzenlileştirmeyi ayarla
            learning_rate=params['learning_rate'],
            eval_metric='mlogloss',
            early_stopping_rounds=params['early_stopping_rounds'],
            enable_categorical=True
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        # Doğrulama seti üzerinde performansı değerlendir
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        self.output_text.append(f"Validation Setinin Doğruluk Oranı: {val_accuracy}")
        self.output_text.append(
            f"Validation Seti Sınıflandırma Raporu:\n{classification_report(y_val, y_val_pred, target_names=le.classes_)}")
        self.output_text.append(f"Validation Seti Confusion Matrix:\n{confusion_matrix(y_val, y_val_pred)}")

        # Test seti üzerinde performansı değerlendir
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.output_text.append(f"Test Setinin Doğruluk Oranı: {accuracy}")
        self.output_text.append(
            f" Test Seti Sınıflandırma Raporu:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
        self.output_text.append(f"Test Seti Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        # ROC Eğrisi için değerler
        y_score = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Grafikleri çiz
        fig = Figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # ROC Eğrisi grafiği
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {0:.2f})'.format(roc_auc))
        ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")

        # Loss grafiği (Eğitim sürecindeki kayıp)
        results = model.evals_result()
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        ax2.plot(x_axis, results['validation_0']['mlogloss'], label='Log Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Log Loss')
        ax2.set_title('XGBoost Training Loss')
        ax2.legend()

        # Grafikleri göster
        canvas = FigureCanvas(fig)
        dialog = QDialog()
        dialog.setWindowTitle('Model Evaluation')
        layout = QVBoxLayout()
        layout.addWidget(canvas)

        # Grafik kaydetme butonu
        save_button = QPushButton('Grafik Kaydet', dialog)
        save_button.clicked.connect(lambda: self.save__figure(fig))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec()

    def save__figure(self, fig):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Grafik Kaydet", "", "PNG Files (*.png);;All Files (*)",
                                                  options=options)
        if fileName:
            fig.savefig(fileName)
            QMessageBox.information
    def apply_train_svm(self):
        dialog = SVMDetailsDialog(self)
        if dialog.exec():
            params = dialog.get_parameters()
            self.output_text.clear()

            # Veri setini yükle
            data = pd.read_csv(self.cleaned_data_filename)
            data.columns = data.columns.str.strip()
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.fillna(0, inplace=True)

            items = data.columns.tolist()
            item, ok = QInputDialog.getItem(self, "Select Target Column", "Choose your target variable:", items, 0,
                                            False)
            if ok and item:
                target_column = item
            else:
                return

            # Kategorik verileri işleme
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].astype('category').cat.codes

            # Label encoding for the target column
            le = LabelEncoder()
            data[target_column] = le.fit_transform(data[target_column])
            label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

            X = data.drop(target_column, axis=1)
            y = data[target_column]
            # Veriyi eğitim, doğrulama ve test setlerine ayır
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                                        random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                              test_size=params['validation_size'], random_state=42)

            # Ölçeklendirici ve SVM modelini birleştir
            smote = SMOTE(random_state=42)
            scaler = StandardScaler()
            svm_model = SVC(C=params['C'], gamma=params['gamma'], max_iter=params['max_iter'],kernel=params['kernel'],degree=params['degree'],coef0=params['coef0'])

            # Pipeline kurulumu
            pipeline = ImPipeline([
                ('smote', smote),
                ('scaler', scaler),
                ('svm', svm_model)
            ])

            # Modeli eğit
            pipeline.fit(X_train, y_train)

            # Doğrulama seti üzerinde performansı değerlendir
            y_val_pred = pipeline.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            self.output_text.append(f"Validation Accuracy: {val_accuracy}\n")
            self.output_text.append(
                f"Classification Report on Validation Set:\n{classification_report(y_val, y_val_pred)}\n")

            # Test seti üzerinde tahmin yapma ve performans metriklerini hesaplama
            y_pred = pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            self.output_text.append(f"Test Set Accuracy: {test_accuracy}")
            self.output_text.append(f"Classification Report on Test Set:\n{classification_report(y_test, y_pred)}")

            # Grafikleri çiz
            self.plot_svm_results(pipeline.named_steps['svm'], X_test, y_test, le.classes_)

    def plot_svm_results(self, model, X_test, y_test, classes):
        try:
            # y_test'i ikili formatına dönüştür
            y_test_binarized = label_binarize(y_test, classes=classes)
            n_classes = y_test_binarized.shape[1]

            # Skorları hesapla
            if hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
            else:
                y_scores = model.predict_proba(X_test)

            # Her sınıf için ROC eğrisi ve AUC hesapla
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Grafikleri çiz
            fig = Figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Her sınıf için ROC eğrisini çiz
            for i in range(n_classes):
                ax1.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Receiver Operating Characteristic')
            ax1.legend(loc="lower right")

            # Doğruluk oranı bar grafiği
            accuracy = accuracy_score(y_test, model.predict(X_test))
            ax2.bar(['Accuracy'], [accuracy])
            ax2.set_title('Model Accuracy')

            # Grafikleri bir QDialog içinde göster
            dialog = QDialog()
            dialog.setWindowTitle("SVM Model Results")
            dialog_layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            dialog_layout.addWidget(canvas)

            # Grafik kaydetme butonu
            save_button = QPushButton('Grafiği Kaydet', dialog)
            save_button.clicked.connect(lambda: self.save____figure(fig))  # Butona işlevsellik kazandır
            dialog_layout.addWidget(save_button)

            dialog.setLayout(dialog_layout)
            dialog.exec_()
        except Exception as e:
            print(f"Error in plotting: {e}")

    def save____figure(self, fig):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self,  # Diyalog penceresi için parent widget
            "Grafiği Kaydet",  # Diyalog başlığı
            "",  # Başlangıç dizini
            "PNG Files (*.png);;All Files (*)",  # Filtre
            options=options
        )
        if fileName:
            fig.savefig(fileName)
            QMessageBox.information(self, "Kaydetme Başarılı", "Grafik Başarıyla kaydedildi")
        else:
            QMessageBox.warning(self, "Kaydetme iptal edildi", "Kaydetme işlevi iptal edildi")

    def apply_knn(self):
        self.output_text.clear()

        dialog = KNNParameterDialog(self)
        if dialog.exec_():
            params = dialog.get_parameters()
        else:
            return  # Kullanıcı iptal ettiyse işlemi durdur

        data_sample = pd.read_csv(self.cleaned_data_filename)
        unique_value_counts = data_sample.nunique()
        single_valued_columns = unique_value_counts[unique_value_counts == 1].index
        data_sample = data_sample.drop(columns=single_valued_columns)
        data_sample.columns = data_sample.columns.str.strip()
        data_sample = data_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

        suitable_columns = data_sample.columns[data_sample.dtypes == 'object'].tolist()
        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:", suitable_columns,
                                                 0, False)
        if not ok or not target_column:
            self.output_text.append("Hedef değişken seçimi iptal edildi.")
            return

        # Hedef değişken için label encoding
        le = LabelEncoder()
        data_sample[target_column] = le.fit_transform(data_sample[target_column])
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        # Diğer tüm kategorik sütunlar için encoding
        for col in data_sample.select_dtypes(include=['object']).columns:
            data_sample[col] = data_sample[col].astype('category').cat.codes

        X = data_sample.drop(columns=[target_column])
        y = data_sample[target_column]

        # Veriyi eğitim, test ve doğrulama setlerine ayırma
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                                    random_state=42)
        val_size_adjusted = params['val_size'] / (1 - params['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted,
                                                          random_state=42)

        # KNN modelini tanımla ve eğit
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'],
                                     metric=params['metric'])
        # K-Fold Cross-Validation uygula
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        accuracies = cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='accuracy')
        model.fit(X_train, y_train)
        self.knn_plot_roc_curve(model, X_test, y_test)  # ROC eğrisini çiz

        # Model değerlendirmesi
        y_pred_val = model.predict(X_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        self.output_text.append(f"Doğrulama Setinin Doğruluk Oranı: {accuracy_val}\n")
        self.output_text.append(f"Doğrulama Seti Sınıflandırma Raporu:\n{classification_report(y_val, y_pred_val,target_names=le.classes_)}")
        self.output_text.append(f"Doğrulama Seti Confusion Matrix:\n{confusion_matrix(y_val, y_pred_val)}\n")

        # Test seti üzerinde tahmin yapma ve performans metriklerini hesaplama
        y_pred_test = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        self.output_text.append(f"Test Setinin Doğruluk Oranı: {accuracy_test}\n")
        self.output_text.append(f"Test Seti Sınıflandırma Raporu:\n{classification_report(y_test, y_pred_test,target_names=le.classes_)}")
        self.output_text.append(f"Test Seti Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

        # Doğruluk grafiği için dialog ve figure oluştur
        accuracy_dialog = QDialog()
        accuracy_dialog.setWindowTitle("KNN Accuracy Over Folds")

        fig_accuracy = Figure(figsize=(10, 8))
        canvas_accuracy = FigureCanvas(fig_accuracy)
        ax_accuracy = fig_accuracy.add_subplot(111)

        # Foldlara göre doğruluk grafiğini çiz
        ax_accuracy.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b')
        ax_accuracy.set_title('KNN Accuracy Over Folds')
        ax_accuracy.set_xlabel('Fold')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.set_xticks(range(1, len(accuracies) + 1))
        ax_accuracy.set_ylim([0, 1])

        layout_accuracy = QVBoxLayout()
        layout_accuracy.addWidget(canvas_accuracy)

        save_button_accuracy = QPushButton('Grafiği Kaydet')
        save_button_accuracy.clicked.connect(lambda: fig_accuracy.savefig(
            QFileDialog.getSaveFileName(accuracy_dialog, 'Grafiği Kaydet',
                                        filter='PNG Files (*.png);;JPG Files (*.jpg)')[0]))
        layout_accuracy.addWidget(save_button_accuracy)

        accuracy_dialog.setLayout(layout_accuracy)
        accuracy_dialog.exec_()
    def knn_plot_roc_curve(self, model, X_test, y_test):
        y_scores = model.predict_proba(X_test)

        # Modeldeki tüm sınıfları listele

        # Eğer model çok sınıflıysa ve 1 sınıfını kontrol etmek istiyorsanız:
        if 1 in model.classes_:
            pos_label_index = list(model.classes_).index(1)
        else:
            # Varsayılan olarak ilk sınıfı kullan (veya uygun bir sınıf seçin)
            pos_label_index = 0  # Bu, sınıflar listesindeki ilk sınıfı kullanır

        # ROC eğrisi için değerleri hesapla
        fpr, tpr, _ = roc_curve(y_test, y_scores[:, pos_label_index], pos_label=model.classes_[pos_label_index])
        roc_auc = auc(fpr, tpr)

        # ROC Eğrisi grafiğini çiz
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")

        # Grafik gösterimini ayarla
        canvas = FigureCanvas(fig)
        dialog = QDialog()
        dialog.setWindowTitle('ROC Curve')
        layout = QVBoxLayout()
        layout.addWidget(canvas)

        # Grafik kaydetme butonu
        save_button = QPushButton('Grafiği Kaydet', dialog)
        save_button.clicked.connect(lambda: self.save____figure(fig))  # Butona işlevsellik kazandır
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def prepare_data(self, data):
        data.columns = data.columns.str.strip()

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)

        # Kategorik verileri sayısallaştır
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            if column != 'Label':
                # Kategorik sütunları uniform bir veri türüne dönüştür
                data[column] = data[column].astype(str)
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                label_encoders[column] = le
        data = data.drop(
            columns=['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
                     'Timestamp', 'SimillarHTTP', 'Inbound'])

        return data

    def graph(self):
        selected_option = None
        for option, checkbox in self.graph_checkboxes.items():
            if checkbox.isChecked():
                selected_option = option
                break

        if selected_option is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir grafik seçeneği seçin.")
            return

        if selected_option == "Random Forest Feature Importance":
            self.show_random_forest_feature_importance_graph()
        elif selected_option == "Korelasyon Matrisi":
            self.show_correlation_matrix_graph()
        elif selected_option == "Aykırı Değer":
            self.show_outlier_graph()
        elif selected_option == "Parametrik random forest":
            self.show_parametric_random_forest_graph()

    def toggle_graph(self):
        self.graph_widget.setVisible(not self.graph_widget.isVisible())

        if self.graph_widget.isVisible():
            self.graph_button.setStyleSheet(self.active_style)
        else:
            self.graph_button.setStyleSheet(self.button_style_menu)

        if self.graph_widget.isVisible():
            # Burada, eğer daha önce "Göster" butonu eklenmemişse, ekleyin
            if not hasattr(self, 'graph_show_button'):
                self.graph_show_button = QPushButton("Göster")
                self.graph_show_button.setStyleSheet("background-color: #1abc9c; color: white;")
                self.graph_show_button.clicked.connect(self.graph)
                self.graph_widget.layout().addWidget(self.graph_show_button)
                setattr(self, 'graph_show_button_added', True)

    def show_random_forest_feature_importance_graph(self):
        if self.clf is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce Random Forest modelini uygulayın.")
            return

        feature_importances = self.clf.feature_importances_
        feature_names = self.X_train_columns
        feature_importances_df = pd.DataFrame({'Features': feature_names, 'Importance': feature_importances})
        feature_importances_df.sort_values(by='Importance', ascending=True, inplace=True)

        # Yeni bir matplotlib figürü ve axes (grafik alanı) oluştur
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(feature_importances_df['Features'], feature_importances_df['Importance'])

        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        ax.set_title('Random Forest Feature Importances')

        # Çubukların üzerine özellik önem değerlerini ekleyerek daha fazla bilgi sağla
        for bar in bars:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}",
                    va='center', ha='left')

        # Figürü bir QDialog içinde göster
        dialog = QDialog()
        dialog.setWindowTitle("Random Forest Feature Importances")
        dialog_layout = QVBoxLayout()
        canvas = FigureCanvas(fig)
        dialog_layout.addWidget(canvas)
        dialog.setLayout(dialog_layout)

        # Opsiyonel: Grafik kaydetme seçeneği
        save_graph = QMessageBox.question(dialog, "Grafik Kaydet", "Grafiği kaydetmek ister misiniz?",
                                          QMessageBox.Yes | QMessageBox.No)
        if save_graph == QMessageBox.Yes:
            file_name, _ = QFileDialog.getSaveFileName(dialog, "Grafik Kaydet", "",
                                                       "PNG Files (*.png);;All Files (*)")
            if file_name:
                fig.savefig(file_name)

        dialog.exec_()  # Dialog'u göster
        plt.close(fig)  # Figürü kapat

    def show_correlation_matrix_graph(self):
        if not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı",
                                "Lütfen önce boşluk doldurma işlemi yapın ve sınıflandırma modelini çalıştırın.")
            return

        try:
            data = pd.read_csv(self.cleaned_data_filename)
            numeric_data = data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_data.corr()

            fig = Figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix')

            canvas = FigureCanvas(fig)
            dialog = QDialog()
            dialog.setWindowTitle("Correlation Matrix")
            dialog_layout = QVBoxLayout()
            dialog_layout.addWidget(canvas)
            dialog.setLayout(dialog_layout)
            dialog.exec_()  # Grafiği göster

            # Grafik kaydetme seçeneği
            save_graph = QMessageBox.question(dialog, "Grafik Kaydet", "Grafiği kaydetmek ister misiniz?",
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if save_graph == QMessageBox.Yes:
                file_name, _ = QFileDialog.getSaveFileName(dialog, "Grafik Kaydet", "",
                                                           "PNG Files (*.png);;All Files (*)")
                if file_name:
                    fig.savefig(file_name)

        except Exception as e:
            self.output_text.append(f"Korelasyon matrisi grafiği oluşturulurken bir hata oluştu: {e}")

    def show_outlier_graph(self):
        if not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı",
                                "Lütfen önce boşluk doldurma işlemi yapın ve sınıflandırma modelini çalıştırın.")
            return

        try:
            data = pd.read_csv(self.cleaned_data_filename)
            fig = Figure(figsize=(15, 7))
            ax = fig.add_subplot(111)
            sns.boxplot(data=data.select_dtypes(include=[np.number]), ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title('Outliers Visualization with Box Plot')

            canvas = FigureCanvas(fig)
            dialog = QDialog()
            dialog.setWindowTitle("Outliers Visualization")
            dialog_layout = QVBoxLayout()
            dialog_layout.addWidget(canvas)
            dialog.setLayout(dialog_layout)
            dialog.exec_()  # Grafiği göster

            # Grafik kaydetme seçeneği
            save_graph = QMessageBox.question(dialog, "Grafik Kaydet", "Grafiği kaydetmek ister misiniz?",
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if save_graph == QMessageBox.Yes:
                file_name, _ = QFileDialog.getSaveFileName(dialog, "Grafik Kaydet", "",
                                                           "PNG Files (*.png);;All Files (*)")
                if file_name:
                    fig.savefig(file_name)

        except Exception as e:
            self.output_text.append(f"Aykırı değerler grafiği oluşturulurken bir hata oluştu: {e}")

    def toggle_feature_selection(self):
        self.feature_selection_widget.setVisible(not self.feature_selection_widget.isVisible())

        if self.feature_selection_widget.isVisible():
            self.feature_selection_button.setStyleSheet(self.active_style)
        else:
            self.feature_selection_button.setStyleSheet(self.button_style_menu)

        if self.feature_selection_widget.isVisible():
            for i in reversed(range(self.feature_selection_widget.layout().count())):
                widget = self.feature_selection_widget.layout().itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() == "SEÇ":
                    widget.deleteLater()

            # Yeni göster tuşunu ekle
            self.show_button = QPushButton("SEÇ")
            self.show_button.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: white;
                border-style: solid;
                border-width: 2px;
                border-radius: 20px;
                border-color: #159b8a;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: 500;
                text-align: center;
                transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
            }
            QPushButton:hover {
                background-color: #17c9b2;
                border-color: #13b3a6;
                transform: scale(1.03);
            }
            QPushButton:pressed {
                background-color: #138d7a;
                border-color: #117a6f;
                transform: scale(0.97);
            }
        """)
            self.show_button.clicked.connect(self.show_feature_selection)
            self.feature_selection_widget.layout().addWidget(self.show_button)

    def show_feature_selection(self):
        selected_methods = [checkbox.text() for checkbox in self.feature_selection_checkboxes.values() if
                            checkbox.isChecked()]

        if 'PCA' in selected_methods:
            self.apply_pca()
        if 'CHI Kare' in selected_methods:
            self.apply_chi_square()
        if 'FISHER' in selected_methods:
            self.apply_fisher()
        if 'F-score' in selected_methods:
            self.apply_fscore()
        if 'Korelasyon tabanlı seçim' in selected_methods:
            self.apply_correlation_based_selection()

    def apply_pca(self):
        self.output_text.clear()
        if not hasattr(self, 'cleaned_data_filename') or not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce boşluk doldurma işlemi yapın.")
            return

        try:
            # Veri setini yükle
            data = pd.read_csv(self.cleaned_data_filename)

            # Anormal verileri temizle
            data = data[~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

            # Sadece sayısal sütunları seç
            numeric_data = data.select_dtypes(include=[np.number])

            # Veriyi standartlaştır
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)

            # Kullanıcıdan PCA için gerekli parametreleri al
            n_components, ok = QInputDialog.getInt(self, "PCA Bileşen Sayısı", "Kullanılacak bileşen sayısını girin:",
                                                   min=1, max=numeric_data.shape[1], step=1)
            if not ok:
                return

            explained_variance, ok = QInputDialog.getDouble(self, "Açıklanan Varyans Oranı",
                                                            "Açıklanacak varyans oranını girin (örn. 0.95):", min=0.0,
                                                            max=1.0, decimals=2)
            if not ok:
                return

            # PCA uygula
            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)

            # Elde edilen veriyi DataFrame'e dönüştür
            pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i + 1}' for i in range(pca_data.shape[1])])

            if pca.explained_variance_ratio_.sum() < explained_variance:
                QMessageBox.warning(self, "Uyarı",
                                    "İstenen varyans oranına ulaşılamadı. Daha fazla bileşen kullanmayı deneyin.")
                return
            # Veriyi kullanıcı arayüzünde göster
            #self.output_text.append(pca_df.head().to_string())

            # Veriyi kaydetme seçeneği
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "PCA Sonuçlarını Kaydet", "", "CSV Dosyaları (*.csv)",
                                                       options=options)
            if file_name:
                pca_df.to_csv(file_name, index=False)
                QMessageBox.information(self, "Başarılı", "PCA sonuçları başarıyla kaydedildi.")

        except Exception as e:
            QMessageBox.warning(self, "Hata", "PCA işlemi sırasında bir hata oluştu: " + str(e))

    def apply_chi_square(self):
        self.output_text.clear()
        if not hasattr(self, 'cleaned_data_filename') or not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce boşluk doldurma işlemi yapın.")
            return

        data = pd.read_csv(self.cleaned_data_filename)

        # Kullanıcıdan hedef değişkeni seçmesini isteyin
        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 data.columns.tolist(), 0, False)
        if not ok or not target_column:
            return  # Kullanıcı iptal ederse veya geçerli bir seçim yapmazsa çık

        # Kategorik sütunları sayısallaştırma işlemi
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            data[column] = data[column].astype(str)  # Tüm metinleri string'e çevir
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])

        # X ve y değerlerini ayarla
        X = data.drop(columns=[target_column])
        y = data[target_column].astype('int')  # Hedef sütunu sayısal değere dönüştür

        # Sonsuz değerleri temizle ve eksik değerleri doldur
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Min-Max ölçeklendirme
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Kullanıcıdan özellik sayısını al
        k, ok = QInputDialog.getInt(self, "Özellik Sayısı", "Kaç özellik seçmek istiyorsunuz?", 1, 1, len(X.columns), 1)
        if not ok:
            return

        # CHI Kare testini uygula
        chi2_selector = SelectKBest(chi2, k=k)
        X_new = chi2_selector.fit_transform(X_scaled, y)

        # Skorları ve p-değerleri ekle
        chi_scores = pd.DataFrame(
            {'Feature': X.columns, 'Chi2 Score': chi2_selector.scores_, 'P-value': chi2_selector.pvalues_})
        chi_scores = chi_scores.sort_values(by='Chi2 Score', ascending=False)

        # Seçilen özelliklerin indekslerini al
        selected_features_indices = chi2_selector.get_support(indices=True)
        selected_features_names = X.columns[selected_features_indices]

        # Sonuçları DataFrame'e dönüştür
        X_selected = pd.DataFrame(X_new, columns=selected_features_names)

        # Sonuçları göster
        #self.output_text.append("Seçilen özellikler:\n" + str(X_selected.head()))

        # Sonuçları bir dosyaya kaydet
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "CHI Kare Sonuçlarını Kaydet", "", "CSV Dosyaları (*.csv)",
                                                   options=options)
        if file_name:
            X_selected.to_csv(file_name, index=False)
            QMessageBox.information(self, "Başarılı", "CHI Kare sonuçları başarıyla kaydedildi.")

    def apply_fisher(self):
        self.output_text.clear()
        if not hasattr(self, 'cleaned_data_filename') or not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce boşluk doldurma işlemi yapın.")
            return

        data = pd.read_csv(self.cleaned_data_filename)

        # Sonsuz değerleri NaN ile değiştir
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN değerleri düşür
        data.dropna(inplace=True)

        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 data.columns[data.dtypes == 'object'].tolist(), 0, False)
        if not ok or not target_column:
            return

        if data[target_column].dtype != 'object':
            QMessageBox.warning(self, "Hata", "Seçilen hedef değişken kategorik bir değişken olmalıdır.")
            return
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Kategorik sütunları Label Encoding ile dönüştürme
        categorical_columns = X.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = X[col].astype(str)  # Kategorik verileri metin türüne dönüştür
            X[col] = le.fit_transform(X[col])

        # Linear Discriminant Analysis (LDA)
        lda = LinearDiscriminantAnalysis()
        try:
            X_new = lda.fit_transform(X, y)
        except ValueError as e:
            QMessageBox.warning(self, "Hata", "Veri dönüşümü sırasında bir hata oluştu: " + str(e))
            return

        # Dönüştürülmüş veriyi DataFrame'e dönüştürme ve gösterme
        X_lda = pd.DataFrame(data=X_new, columns=[f'LD{i + 1}' for i in range(X_new.shape[1])])
        self.output_text.append(X_lda.head().to_string())

        # Sonuçları kaydetme
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Fisher LDA Sonuçlarını Kaydet", "",
                                                   "CSV Dosyaları (*.csv)", options=options)
        if file_name:
            X_lda.to_csv(file_name, index=False)
            QMessageBox.information(self, "Başarılı", "Fisher LDA sonuçları başarıyla kaydedildi.")

    def apply_fscore(self):
        self.output_text.clear()
        if not hasattr(self, 'cleaned_data_filename') or not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce boşluk doldurma işlemi yapın.")
            return

        # Veri setini yükleyin
        data = pd.read_csv(self.cleaned_data_filename)

        # Sonsuz değerleri NaN ile değiştir
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Non-numeric veriler için Label Encoding uygula
        label_encoders = {}
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

        # NaN değerleri ortalama ile doldurun
        imputer = SimpleImputer(strategy='mean')
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

        # Hedef değişkenin sürekli olup olmadığını kontrol et ve kullanıcıya seçim yapmasını sağla
        continuous_columns = data.select_dtypes(include=[np.float64, np.int64]).columns
        target_column, ok = QInputDialog.getItem(self, "Hedef Değişken Seç", "Hedef değişkeni seçin:",
                                                 continuous_columns.tolist(), 0, False)
        if not ok or not target_column:
            return

        # Bağımsız ve bağımlı değişkenleri ayır
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Kullanıcıdan özellik sayısını al
        k, ok = QInputDialog.getInt(self, "Özellik Sayısı", "Kaç özellik seçmek istiyorsunuz?", min=1, max=X.shape[1],
                                    step=1, value=5)
        if not ok:
            return

        # F-Skor ile özellik seçme modelini oluştur
        f_score_selector = SelectKBest(f_classif, k=k)
        X_new = f_score_selector.fit_transform(X, y)

        # Seçilen özelliklerin indekslerini al
        selected_features_indices = f_score_selector.get_support(indices=True)
        selected_features_names = X.columns[selected_features_indices]

        # Yeni veri setini DataFrame olarak oluştur
        X_selected = pd.DataFrame(X_new, columns=selected_features_names)

        # Sonuçları arayüzde göster
        self.output_text.append("Seçilen en iyi özellikler:")
        self.output_text.append(str(X_selected.columns.tolist()))

        # Veriyi kaydetme seçeneği
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "F-Score Sonuçlarını Kaydet", "", "CSV Dosyaları (*.csv)",
                                                   options=options)
        if file_name:
            X_selected.to_csv(file_name, index=False)
            QMessageBox.information(self, "Başarılı", "F-Score sonuçları başarıyla kaydedildi.")

    def apply_correlation_based_selection(self):
        self.output_text.clear()
        if not hasattr(self, 'cleaned_data_filename') or not self.cleaned_data_filename:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce boşluk doldurma işlemi yapın.")
            return

        data = pd.read_csv(self.cleaned_data_filename)

        # Metinsel veya kategorik özellikleri Label Encoding ile dönüştürün
        label_encoders = {}
        categorical_columns = data.select_dtypes(include=['object']).columns

        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le

        # Korelasyon matrisini hesaplayın
        correlation_matrix = data.corr()

        threshold, ok = QInputDialog.getDouble(
            self, "Korelasyon Eşik Değeri",
            "Korelasyon eşik değeri girin (örneğin, 0.6):",
            0.6, 0.0, 1.0, 2
        )
        if not ok:
            return

        # Yüksek korelasyona sahip özellikleri seçin
        highly_correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_features.add(colname)

        # Yüksek korelasyona sahip özellikleri veri çerçevesinden kaldırın
        son_hali_filtered = data.drop(columns=highly_correlated_features)

        # Sonuçları arayüzde göster
        self.output_text.append("Korelasyon eşik değerine göre kaldırılan özellikler:")
        self.output_text.append(str(highly_correlated_features))

        # Veriyi kaydetme seçeneği
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Korelasyon Tabanlı Seçim Sonuçlarını Kaydet", "",
                                                   "CSV Dosyaları (*.csv)", options=options)
        if file_name:
            son_hali_filtered.to_csv(file_name, index=False)
            QMessageBox.information(self, "Başarılı",
                                    f"Filtrelenmiş veri çerçevesi {file_name} dosyasına başarıyla kaydedildi.")

    def toggle_explain_ai(self):
        self.explain_ai_widget.setVisible(not self.explain_ai_widget.isVisible())

        if self.explain_ai_widget.isVisible():
            self.explain_ai_button.setStyleSheet(self.active_style)
        else:
            self.explain_ai_button.setStyleSheet(self.button_style_menu)

        if self.explain_ai_widget.isVisible():
            for i in reversed(range(self.explain_ai_widget.layout().count())):
                widget = self.explain_ai_widget.layout().itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() == "HESAPLA":
                    widget.deleteLater()

            # Yeni göster tuşunu ekle
            self.show_button = QPushButton("HESAPLA")
            self.show_button.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: white;
                border-style: solid;
                border-width: 2px;
                border-radius: 20px;
                border-color: #159b8a;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: 500;
                text-align: center;
                transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
            }
            QPushButton:hover {
                background-color: #17c9b2;
                border-color: #13b3a6;
                transform: scale(1.03);
            }
            QPushButton:pressed {
                background-color: #138d7a;
                border-color: #117a6f;
                transform: scale(0.97);
            }
        """)
            self.show_button.clicked.connect(self.show_explain_ai)
            self.explain_ai_widget.layout().addWidget(self.show_button)

    def show_explain_ai(self):
        selected_methods = [checkbox.text() for checkbox in self.explain_ai_checkboxes.values() if checkbox.isChecked()]

        if 'SHAP' in selected_methods:
            self.perform_shap_analysis()
        if 'LIME' in selected_methods:
            self.apply_lime()

    def perform_shap_analysis(self):
        if self.clf is None or not hasattr(self, 'X_test'):
            QMessageBox.warning(self, "Uyarı",
                                "Lütfen önce Random Forest modelini eğitin ve test veri setini belirleyin.")
            return

        try:
            explainer = shap.TreeExplainer(self.clf)
            shap_values = explainer.shap_values(self.X_test.sample(100))



            # SHAP özeti grafiğini bir Figure içinde oluştur
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, self.X_test.sample(100), plot_type="bar", feature_names=self.X_train_columns,class_names=self.le.classes_,
                              show=False)
            canvas = FigureCanvas(fig)

            # Grafiği göstermek için bir QDialog oluştur
            dialog = QDialog(self)
            dialog.setWindowTitle("SHAP Özet Grafiği")
            layout = QVBoxLayout(dialog)

            # Grafiği QDialog'a ekleyin
            layout.addWidget(canvas)

            # Kaydetme butonu
            save_btn = QPushButton("Grafiği Kaydet", dialog)
            save_btn.clicked.connect(lambda: self.save_plot(fig))
            layout.addWidget(save_btn)

            dialog.exec_()  # Dialog'u çalıştır

        except Exception as e:
            QMessageBox.warning(self, "Hata", f"SHAP analizi sırasında bir hata oluştu: {str(e)}")

    def save_plot(self, fig):
        file_name, _ = QFileDialog.getSaveFileName(self, "Grafiği Kaydet", "", "PNG Files (*.png);;All Files (*)")
        if file_name:
            fig.savefig(file_name)
            QMessageBox.information(self, "Başarılı", "Grafik başarıyla kaydedildi.")
    def apply_lime(self):
        self.output_text.clear()
        if self.clf is None or not hasattr(self, 'X_test'):
            QMessageBox.warning(self, "Uyarı",
                                "Lütfen önce Random Forest modelini eğitin ve test veri setini belirleyin.")
            return
            # Mevcut çıktıyı sakla
        current_text = self.output_text.toPlainText()
        try:
            # Lime için özellik sayısı
            num_features, ok = QInputDialog.getInt(self, "Özellik Sayısı",
                                                   "Açıklanacak özellik sayısını girin (örn. 5):", min=1, max=20,
                                                   step=1, value=5)
            if not ok:
                return

            # LIME açıklamasını yap
            explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train.values,
                                                               feature_names=self.X_train.columns.tolist(),
                                                               class_names=['Negative', 'Positive'], verbose=True,
                                                               mode='classification')

            # Random bir örnek seç
            i = np.random.randint(0, self.X_test.shape[0])
            exp = explainer.explain_instance(self.X_test.values[i], self.clf.predict_proba, num_features=num_features)

            # Açıklama sonuçlarını al
            explanation = exp.as_list()
            features, effects = zip(*explanation)

            # Özellikler ve etkileri yazdırma
            self.output_text.append("Özellikler ve Etkileri:")
            for feature, effect in zip(features, effects):
                self.output_text.append(f"{feature}: {effect:.2f}")
            # Bar grafiği oluştur
            fig, ax = plt.subplots(figsize=(8, 8))  # Grafik boyutunu ayarla
            y_pos = np.arange(len(features))
            ax.barh(y_pos, effects, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)  # Etiket boyutunu ayarla
            ax.invert_yaxis()  # Etiketlerin üstten aşağıya doğru okunmasını sağlar
            ax.set_xlabel('Effect')
            ax.set_title('LIME Açıklaması')
            fig.subplots_adjust(left=0.4)  # Y eksenindeki etiketler için daha fazla yer sağla

            # Grafiği göstermek için bir QDialog oluştur
            canvas = FigureCanvas(fig)
            dialog = QDialog(self)
            dialog.setWindowTitle("LIME Açıklaması")
            layout = QVBoxLayout(dialog)
            layout.addWidget(canvas)

            # Grafik kaydetme butonu
            save_button = QPushButton("Grafiği Kaydet", dialog)
            save_button.clicked.connect(lambda: self.save_plot(fig))
            layout.addWidget(save_button)

            # Mevcut çıktıyı geri yükle ve sonuçları ekle
            self.output_text.setPlainText(current_text + '\n' + f"Model explained for instance {i}:\n{exp.as_list()}")

            dialog.exec_()  # Dialog'u çalıştır
            plt.close(fig)  # Kullanılmayan figürü kapat

        except Exception as e:
            QMessageBox.warning(self, "Hata", f"LIME analizi sırasında bir hata oluştu: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())