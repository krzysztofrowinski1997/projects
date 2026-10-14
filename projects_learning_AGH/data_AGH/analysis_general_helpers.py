# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 17:30:45 2025

@author: g.migut

Biblioteka pomocniczych funkcji przydatnych podczas wstępnej eksploracji danych
"""

# Importujemy biblioteki do obróbki danych, wizualizacji[ i budowy modeli]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import _tree

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def plot_box_strip(df, variables, hue=None):
    """
    Tworzy wykresy pudełkowe z nakładającymi się wykresami strip dla podanych zmiennych.
    
    Parametry:
    - df: DataFrame zawierający dane
    - variables: lista nazw kolumn dla zmiennych ilościowych
    - hue: opcjonalna zmienna grupująca (kategoryczna)
    """
    for var in variables:
        plt.figure(figsize=(8, 6))
        
        # Wykres pudełkowy
        sns.boxplot(data=df, y=var, x=hue, fliersize=0, color='green')

        # Wykres strip
        sns.stripplot(data=df, y=var, x=hue, jitter=0.1, alpha=0.3, color='k', dodge=True if hue else False)
        
        plt.title(f'Wykres pudełkowy: {var}')
        plt.xlabel(hue if hue else '')
        plt.ylabel(var)
        plt.show()
        
        

# graficzna analiza rozkładu zmiennej jakościowej
def plot_categorical_bars(df, categorical_vars):
    """
    Tworzy wykresy słupkowe dla każdej zmiennej jakościowej w podanej ramce danych.

    Parametry:
    - df: DataFrame zawierający zmienne jakościowe
    - categorical_vars: lista nazw kolumn zawierających zmienne jakościowe
    """
    for var in categorical_vars:
        plt.figure(figsize=(6, 4))
        df[var].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Liczność kategorii dla zmiennej: {var}')
        plt.xlabel(var)
        plt.ylabel('Liczność')
        plt.xticks(rotation=45)  # Obrót etykiet kategorii dla czytelności
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        
def calculate_iv(df, y, num_vars, cat_vars, n_bins=10):
    """
    Oblicza Information Value (IV) dla zmiennych ilościowych i jakościowych.

    Parametry:
    - df: DataFrame z danymi
    - y: nazwa kolumny ze zmienną zależną (binarna: 0/1)
    - num_vars: lista zmiennych ilościowych
    - cat_vars: lista zmiennych jakościowych
    - n_bins: liczba przedziałów do dyskretyzacji zmiennych ilościowych (domyślnie 10)

    Zwraca:
    - DataFrame z kolumnami ['zmienna', 'IV'], posortowany malejąco
    """

    def compute_iv_table(grouped):
        # udział klasy 1 i 0
        dist_good = grouped[1] / grouped[1].sum()
        dist_bad = grouped[0] / grouped[0].sum()
        woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))  # dodajemy małą stałą, by uniknąć dzielenia przez zero
        iv = ((dist_good - dist_bad) * woe).sum()
        return iv

    iv_list = []

    for var in num_vars:
        try:
            binned = pd.qcut(df[var], q=n_bins, duplicates='drop')
            grouped = df.groupby([binned, df[y]]).size().unstack(fill_value=0)
            iv = compute_iv_table(grouped)
            iv_list.append({'zmienna': var, 'IV': iv})
        except Exception as e:
            iv_list.append({'zmienna': var, 'IV': np.nan})

    for var in cat_vars:
        try:
            grouped = df.groupby([df[var], df[y]]).size().unstack(fill_value=0)
            iv = compute_iv_table(grouped)
            iv_list.append({'zmienna': var, 'IV': iv})
        except Exception as e:
            iv_list.append({'zmienna': var, 'IV': np.nan})

    iv_df = pd.DataFrame(iv_list).sort_values(by='IV', ascending=False).reset_index(drop=True)
    return iv_df


def calculate_iv_new(df, y, num_vars, cat_vars, n_bins=10):
    """
    Oblicza Information Value (IV) dla zmiennych ilościowych i jakościowych.

    Parametry:
    - df: DataFrame z danymi
    - y: nazwa kolumny ze zmienną zależną (binarna: 0/1)
    - num_vars: lista zmiennych ilościowych
    - cat_vars: lista zmiennych jakościowych
    - n_bins: liczba przedziałów do dyskretyzacji zmiennych ilościowych (domyślnie 10)

    Zwraca:
    - DataFrame z kolumnami ['zmienna', 'IV', 'siła_predykcyjna'], posortowany malejąco
    """

    def compute_iv_table(grouped):
        dist_good = grouped[1] / grouped[1].sum()
        dist_bad = grouped[0] / grouped[0].sum()
        woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
        iv = ((dist_good - dist_bad) * woe).sum()
        return iv

    def interpret_iv(iv):
        if iv < 0.02:
            return "brak wartości predykcyjnej"
        elif iv < 0.1:
            return "słaba"
        elif iv < 0.3:
            return "średnia"
        elif iv < 0.5:
            return "silna"
        else:
            return "bardzo silna / podejrzana"

    iv_list = []

    for var in num_vars:
        try:
            binned = pd.qcut(df[var], q=n_bins, duplicates='drop')
            grouped = df.groupby([binned, df[y]]).size().unstack(fill_value=0)
            iv = compute_iv_table(grouped)
            iv_list.append({'zmienna': var, 'IV': iv, 'siła_predykcyjna': interpret_iv(iv)})
        except Exception as e:
            iv_list.append({'zmienna': var, 'IV': np.nan, 'siła_predykcyjna': 'błąd'})

    for var in cat_vars:
        try:
            grouped = df.groupby([df[var], df[y]]).size().unstack(fill_value=0)
            iv = compute_iv_table(grouped)
            iv_list.append({'zmienna': var, 'IV': iv, 'siła_predykcyjna': interpret_iv(iv)})
        except Exception as e:
            iv_list.append({'zmienna': var, 'IV': np.nan, 'siła_predykcyjna': 'błąd'})

    iv_df = pd.DataFrame(iv_list).sort_values(by='IV', ascending=False).reset_index(drop=True)
    return iv_df



def plot_woe_charts(df, y, num_vars, cat_vars, bad_class=1, n_bins=10):
    """
    Rysuje wykresy WOE dla zmiennych ilościowych i jakościowych.

    Parametry:
    - df: DataFrame
    - y: nazwa zmiennej zależnej (binarnej)
    - num_vars: lista zmiennych ilościowych
    - cat_vars: lista zmiennych jakościowych
    - bad_class: wartość oznaczająca klasę negatywną (domyślnie: 1)
    - n_bins: liczba przedziałów dla zmiennych ilościowych
    """

    def compute_woe_table(grouped):
        # dodajemy małą stałą do proporcji, by uniknąć dzielenia przez zero
        good = grouped[~grouped.index.get_level_values(1).isin([bad_class])]
        bad = grouped[grouped.index.get_level_values(1).isin([bad_class])]

        dist_good = good.groupby(level=0).sum() / good.sum().sum()
        dist_bad = bad.groupby(level=0).sum() / bad.sum().sum()

        woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
        return woe

    def plot_woe(woe_series, varname):
        plt.figure(figsize=(8, 5))
        
        sns.barplot(
            x=woe_series.index.astype(str),
            y=woe_series.values,
            color="steelblue"  # stały kolor dla wszystkich słupków
            )
        
        #sns.barplot(x=woe_series.index.astype(str), y=woe_series.values, palette="Blues")
        
        plt.title(f"Wykres WoE {varname}", fontsize=14)
        plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
        plt.ylabel("Weight of Evidence")
        plt.xlabel(varname)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        
    def plot_woe_new(woe_series, dist_good, dist_bad, varname):
        # Oblicz wkład do IV dla każdego binu
        iv_bins = (dist_good - dist_bad) * woe_series
    
        # Normalizacja wartości IV_bin do zakresu 0–1
        iv_scaled = (iv_bins - iv_bins.min()) / (iv_bins.max() - iv_bins.min() + 1e-6)
    
        # Stworzenie mapy kolorów (ciemniejszy = większy wkład do IV)
        cmap = sns.color_palette("Blues", as_cmap=True)
        colors = [cmap(val) for val in iv_scaled]
    
        # Rysowanie wykresu
        plt.figure(figsize=(8, 5))
        sns.barplot(x=woe_series.index.astype(str), y=woe_series.values, palette=colors)
        plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
        plt.title(f"Wykres WoE – {varname}", fontsize=14)
        plt.ylabel("Weight of Evidence")
        plt.xlabel(varname)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()


    # zmienne ilościowe
    for var in num_vars:
        try:
            binned = pd.qcut(df[var], q=n_bins, duplicates='drop')
            grouped = df.groupby([binned, df[y]]).size()
            woe = compute_woe_table(grouped)
            plot_woe(woe, var)
        except Exception as e:
            print(f"Nie udało się wygenerować WOE dla zmiennej ilościowej: {var}\n{e}")

    # zmienne jakościowe
    for var in cat_vars:
        try:
            grouped = df.groupby([df[var], df[y]]).size()
            woe = compute_woe_table(grouped)
            plot_woe(woe, var)
        except Exception as e:
            print(f"Nie udało się wygenerować WOE dla zmiennej jakościowej: {var}\n{e}")


def group_rare_categories(df, col, threshold, other_label="other"):
    """
    Grupuje rzadkie kategorie w zmiennej jakościowej do jednej wspólnej etykiety.

    Parametry:
    - df: DataFrame
    - col: nazwa kolumny kategorycznej
    - threshold: maksymalna liczba wystąpień, poniżej której kategorie zostaną zgrupowane
    - other_label: etykieta przypisana rzadkim kategoriom (domyślnie 'other')

    Zwraca:
    - Series z przekształconymi kategoriami
    """
    value_counts = df[col].value_counts()
    rare_categories = value_counts[value_counts <= threshold].index
    return df[col].apply(lambda x: other_label if x in rare_categories else x)


def get_order_by_woe(df, target, feature):
    """
    Zwraca listę kategorii zmiennej kategorycznej posortowaną wg wartości WoE.

    Parametry:
    - df: DataFrame
    - target: nazwa zmiennej zależnej (binarna: 0/1)
    - feature: nazwa zmiennej kategorycznej

    Zwraca:
    - Lista kategorii posortowana rosnąco wg WoE
    """
    grouped = df.groupby([feature, df[target]]).size().unstack(fill_value=0)

    # Dodanie małej stałej, by uniknąć dzielenia przez 0
    dist_good = grouped[1] / grouped[1].sum()
    dist_bad = grouped[0] / grouped[0].sum()
    woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))

    # Posortowanie po WOE rosnąco
    return woe.sort_values().index.tolist()


def encode_ordinal(df, column, ordered_categories, new_column=None):
    """
    Koduje zmienną jakościową zgodnie z porządkiem kategorii (ordinal encoding).

    Parametry:
    - df: DataFrame
    - column: nazwa kolumny do zakodowania
    - ordered_categories: lista kategorii w kolejności od najmniejszej do największej
    - new_column: nazwa nowej kolumny (jeśli None, nadpisuje oryginalną)

    Zwraca:
    - DataFrame z zakodowaną kolumną
    """
    mapping = {cat: i for i, cat in enumerate(ordered_categories)}
    target_col = new_column if new_column is not None else column
    df[target_col] = df[column].map(mapping)
    return df


def encode_multiple_ordinal_by_woe(df, target, cat_vars):
    """
    Dla listy zmiennych jakościowych wykonuje ordinal encoding na podstawie wartości WoE,
    nadpisując oryginalne kolumny.

    Parametry:
    - df: DataFrame wejściowy
    - target: nazwa zmiennej zależnej binarnej
    - cat_vars: lista zmiennych jakościowych do zakodowania

    Zwraca:
    - DataFrame z zakodowanymi zmiennymi (nadpisane kolumny)
    """
    df_copy = df.copy()

    for col in cat_vars:
        try:
            order = get_order_by_woe(df_copy, target, col)
            df_copy = encode_ordinal(df_copy, column=col, ordered_categories=order)
        except Exception as e:
            print(f"⚠️ Błąd kodowania zmiennej '{col}': {e}")

    return df_copy




def plot_roc_curve(y_true, y_score, positive_label, title="Krzywa ROC"):
    """
    Rysuje krzywą ROC dla podanych etykiet i wyników predykcji.

    Parametry:
    - y_true: rzeczywiste wartości zmiennej zależnej
    - y_score: przewidywane prawdopodobieństwa klasy pozytywnej
    - positive_label: wartość klasy pozytywnej (dla której liczona jest czułość)
    - title: tytuł wykresu (opcjonalny)
    """
    # Zamiana etykiet na 0/1 względem klasy pozytywnej
    y_binary = [1 if y == positive_label else 0 for y in y_true]

    # Wyznaczenie krzywej ROC
    fpr, tpr, thresholds = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)

    # Rysowanie wykresu
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # linia losowa
    plt.xlabel('False Positive Rate (1 - specyficzność)')
    plt.ylabel('True Positive Rate (czułość)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_roc_curve_compare(y_true, y_score_main, positive_label,
                           y_score_cv=None, label_main="Model", label_cv="Cross-val", title="Krzywa ROC"):
    """
    Rysuje krzywą ROC dla jednego lub dwóch zestawów predykcji.

    Parametry:
    - y_true: rzeczywiste klasy
    - y_score_main: predykcje modelu głównego (prawdopodobieństwa)
    - positive_label: wartość klasy pozytywnej
    - y_score_cv: (opcjonalnie) drugi zestaw predykcji, np. z cross-validation
    - label_main: etykieta dla modelu głównego
    - label_cv: etykieta dla drugiego modelu (cv)
    - title: tytuł wykresu
    """
    # Binarna wersja y_true (czy dana obserwacja to klasa pozytywna?)
    y_bin = [1 if y == positive_label else 0 for y in y_true]

    # ROC – model główny
    fpr_main, tpr_main, _ = roc_curve(y_bin, y_score_main)
    auc_main = auc(fpr_main, tpr_main)

    # Wykres
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_main, tpr_main, label=f"{label_main} (AUC = {auc_main:.3f})", linewidth=2)

    # ROC – drugi zestaw (opcjonalny)
    if y_score_cv is not None:
        fpr_cv, tpr_cv, _ = roc_curve(y_bin, y_score_cv)
        auc_cv = auc(fpr_cv, tpr_cv)
        plt.plot(fpr_cv, tpr_cv, label=f"{label_cv} (AUC = {auc_cv:.3f})", linewidth=2, linestyle='--')

    # Linia losowa
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)

    # Etykiety
    plt.xlabel("False Positive Rate (1 - specyficzność)")
    plt.ylabel("True Positive Rate (czułość)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_roc_curve_compare_multi(y_true, pred_dict, positive_label, title="Krzywa ROC – porównanie modeli"):
    """
    Rysuje krzywe ROC dla wielu zestawów predykcji.

    Parametry:
    - y_true: rzeczywiste etykiety klas
    - pred_dict: słownik w formacie {nazwa_modelu: predykcje_prawdopodobieństwa}
    - positive_label: etykieta klasy pozytywnej
    - title: tytuł wykresu
    """
    y_bin = [1 if y == positive_label else 0 for y in y_true]

    plt.figure(figsize=(7, 6))

    for model_name, y_score in pred_dict.items():
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate (1 - specyficzność)")
    plt.ylabel("True Positive Rate (czułość)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()



def evaluate_auc_ks_model_vs_cv(y_true, y_pred_model, y_pred_cv, positive_label=1):
    """
    Porównuje AUC i KS (oraz próg decyzyjny KS) dla modelu klasycznego i cross-val (out-of-fold).

    Parametry:
    - y_true: rzeczywiste etykiety klas
    - y_pred_model: predykcje prawdopodobieństw z modelu trenowanego na całości
    - y_pred_cv: predykcje prawdopodobieństw z cross-validation
    - positive_label: etykieta klasy pozytywnej

    Zwraca:
    - DataFrame z AUC, KS oraz progiem decyzyjnym dla KS
    """
    y_bin = [1 if y == positive_label else 0 for y in y_true]

    def ks_stat(y_true_bin, y_score):
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
        ks_values = np.abs(tpr - fpr)
        ks = ks_values.max()
        best_threshold = thresholds[ks_values.argmax()]
        return ks, best_threshold

    # Model klasyczny
    auc_model = roc_auc_score(y_bin, y_pred_model)
    ks_model, ks_thr_model = ks_stat(y_bin, y_pred_model)

    # Cross-val
    auc_cv = roc_auc_score(y_bin, y_pred_cv)
    ks_cv, ks_thr_cv = ks_stat(y_bin, y_pred_cv)

    # Tabelka wyników
    df = pd.DataFrame({
        'Podejście': ['Model klasyczny', 'Cross-validation'],
        'AUC': [auc_model, auc_cv],
        'KS': [ks_model, ks_cv],
        'Threshold_KS': [ks_thr_model, ks_thr_cv]
    })

    return df





def extract_tree_rules(model, feature_names, class_names=None, decimals=2):
    """
    Ekstrahuje reguły decyzyjne z dopasowanego drzewa i zapisuje je w czytelnej tabeli.

    Parametry:
    - model: dopasowany DecisionTreeClassifier
    - feature_names: lista nazw cech (w takiej samej kolejności jak w X)
    - class_names: lista etykiet klas (opcjonalnie, np. ["Nie przeżył", "Przeżył"])
    - decimals: liczba miejsc po przecinku w prawdopodobieństwach

    Zwraca:
    - DataFrame z kolumnami: ['Reguła', 'Liczność', 'P(1)', 'P(0)']
    """
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name} ≤ {threshold:.2f}"])
            recurse(tree_.children_right[node], path + [f"{name} > {threshold:.2f}"])
        else:
            proba = tree_.value[node][0] / tree_.value[node][0].sum()
            paths.append({
                'Reguła': " AND ".join(path),
                'Liczność': int(tree_.n_node_samples[node]),
                'P(1)': round(proba[1], decimals),
                'P(0)': round(proba[0], decimals)
            })

    recurse(0, [])

    return pd.DataFrame(paths).sort_values(by="P(1)", ascending=False).reset_index(drop=True)




import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def basic_metrics_summary(y_true, y_pred_proba, positive_label=1, threshold=0.5, show_formula=True):
    """
    Oblicza podstawowe miary jakości klasyfikacji binarnej, z opcją wyświetlenia wzorów.

    Parametry:
    - y_true: rzeczywiste etykiety (0/1)
    - y_pred_proba: przewidywane prawdopodobieństwa klasy pozytywnej
    - positive_label: wartość klasy pozytywnej (default = 1)
    - threshold: próg odcięcia (default = 0.5)
    - show_formula: czy pokazać kolumnę ze wzorami (default = True)

    Zwraca:
    - DataFrame z metrykami: skrót, nazwa polska, wartość, (opcjonalnie) wzór
    """

    y_bin = np.array([1 if y == positive_label else 0 for y in y_true])
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred).ravel()

    # Metryki
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc  = (tp + tn) / (tp + tn + fp + fn)
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else np.nan

    # Dane do tabeli
    data = {
        "Skrót": ["SENS", "SPEC", "PREC", "NPV", "ACC", "F1"],
        "Miara (PL)": [
            "Czułość (Sensitivity)",
            "Specyficzność (Specificity)",
            "Precyzja (Precision)",
            "Ujemna wartość predykcyjna (NPV)",
            "Dokładność (Accuracy)",
            "F1-score (średnia precyzji i czułości)"
        ],
        "Wartość": [sens, spec, prec, npv, acc, f1]
    }

    if show_formula:
        data["Wzór"] = [
            "TP / (TP + FN)",
            "TN / (TN + FP)",
            "TP / (TP + FP)",
            "TN / (TN + FN)",
            "(TP + TN) / (TP + TN + FP + FN)",
            "2 * PREC * SENS / (PREC + SENS)"
        ]

    return pd.DataFrame(data).round(4)




def tree_model_importance(model, feature_names=None, top_n=None, title="Ważność cech w modelu drzewa"):
    """
    Tworzy wykres poziomych słupków z ważnością cech w DecisionTreeClassifier.

    Parametry:
    - model: dopasowany obiekt DecisionTreeClassifier
    - feature_names: lista nazw cech (jeśli None, użyje indeksów cech)
    - top_n: ile najważniejszych cech pokazać (jeśli None, pokazuje wszystkie)
    - title: tytuł wykresu
    """
    # Nazwy cech
    if feature_names is None:
        feature_names = [f"Cech{idx}" for idx in range(len(model.feature_importances_))]

    # Tabela z ważnościami
    importance_df = pd.DataFrame({
        'Cechy': feature_names,
        'Ważność': model.feature_importances_
    }).sort_values(by='Ważność', ascending=True)

    # Ograniczenie do top_n (opcjonalnie)
    if top_n is not None:
        importance_df = importance_df.tail(top_n)

    # Wykres
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['Cechy'], importance_df['Ważność'], color='skyblue')
    plt.xlabel("Ważność")
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_category_distribution_percent(df, group_col, category_col, figsize=(10, 5)):
    """
    Rysuje procentowy wykres słupkowy pokazujący udział kategorii w każdej grupie.

    Parametry:
        df           : DataFrame z danymi
        group_col    : kolumna z numerem grupy/klastra
        category_col : kolumna z kategoriami do porównania
        figsize      : rozmiar wykresu
    """
    counts = df.groupby([group_col, category_col]).size().reset_index(name='Liczba')
    total = counts.groupby(group_col)['Liczba'].transform('sum')
    counts['Procent'] = 100 * counts['Liczba'] / total

    plt.figure(figsize=figsize)
    sns.barplot(data=counts, x=group_col, y='Procent', hue=category_col)
    plt.title(f"Procentowy rozkład '{category_col}' w grupach '{group_col}'")
    plt.xlabel(group_col)
    plt.ylabel("Udział procentowy")
    plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

################################
## Logistyczna
################################    
    
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


def fit_logit_dual(X_train, y_train):
    """
    Buduje model logistyczny w statsmodels (dla interpretacji) oraz sklearn (dla predykcji).
    Zwraca: DataFrame z OR i p-value oraz model sklearn.
    """

    # --- statsmodels (do interpretacji) ---
    X_const = sm.add_constant(X_train)
    sm_model = sm.Logit(y_train, X_const).fit(disp=0)

    coef = sm_model.params
    or_values = np.exp(coef)
    p_values = sm_model.pvalues

    interpretation = []
    for or_val in or_values:
        if or_val > 1.0001:
            interpretation.append("Zwiększa szanse")
        elif or_val < 0.9999:
            interpretation.append("Zmniejsza szanse")
        else:
            interpretation.append("Brak istotnego wpływu")

    summary_df = pd.DataFrame({
        "Zmienna": coef.index,
        "Współczynnik (log-odds)": coef.round(4),
        "Iloraz szans (OR)": or_values.round(4),
        "p-value": p_values.round(4),
        "Interpretacja": interpretation
    })

    # --- sklearn (do predykcji) ---
    skl_model = LogisticRegression(solver='liblinear')
    skl_model.fit(X_train, y_train)

    return summary_df, skl_model



def backward_selection(X, y, threshold_out=0.05):
    X_ = X.copy()
    while True:
        X_const = sm.add_constant(X_)
        model = sm.Logit(y, X_const).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]  # pomijamy intercept
        max_p = pvalues.max()
        if max_p > threshold_out:
            worst_feature = pvalues.idxmax()
            X_ = X_.drop(columns=[worst_feature])
        else:
            break
    return X_

def stepwise_logit_summary(X_train, y_train):
    """
    Przeprowadza selekcję zmiennych metodą krokową wsteczną.
    Tworzy model statsmodels i sklearn na wybranych zmiennych.
    Zwraca: ramkę z OR, model sklearn i listę wybranych zmiennych.
    """

    # Selekcja zmiennych
    X_selected = backward_selection(X_train, y_train)

    # --- statsmodels ---
    X_const = sm.add_constant(X_selected)
    sm_model = sm.Logit(y_train, X_const).fit(disp=0)

    coef = sm_model.params
    or_values = np.exp(coef)
    p_values = sm_model.pvalues

    interpretation = []
    for or_val in or_values:
        if or_val > 1.0001:
            interpretation.append("Zwiększa szanse")
        elif or_val < 0.9999:
            interpretation.append("Zmniejsza szanse")
        else:
            interpretation.append("Brak istotnego wpływu")

    summary_df = pd.DataFrame({
        "Zmienna": coef.index,
        "Współczynnik (log-odds)": coef.round(4),
        "Iloraz szans (OR)": or_values.round(4),
        "p-value": p_values.round(4),
        "Interpretacja": interpretation
    })

    # --- sklearn ---
    skl_model = LogisticRegression(solver='liblinear')
    skl_model.fit(X_selected, y_train)

    return summary_df, skl_model, list(X_selected.columns)




from sklearn.metrics import confusion_matrix
import pandas as pd

def confusion_matrix_with_labels(y_true, y_pred_proba, threshold=0.5, margins=True):
    """
    Zwraca opisaną macierz pomyłek z oznaczeniami TP/FP/FN/TN
    oraz opcjonalnymi sumami brzegowymi.

    Parametry:
    - y_true: rzeczywiste klasy (0 lub 1)
    - y_pred_proba: przewidywane prawdopodobieństwa klasy pozytywnej
    - threshold: punkt odcięcia (default = 0.5)
    - margins: czy dodać sumy brzegowe (default = True)

    Zwraca:
    - DataFrame z opisem i oznaczeniami TP/FP/FN/TN
    """

    # Przekształcenie do postaci binarnej
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Macierz pomyłek (predykcja jako wiersze, rzeczywiste jako kolumny)
    cm = confusion_matrix(y_true, y_pred).T  # transpozycja

    # Poprawne etykiety:
    # wiersze: Przewidywane TAK/NIE
    # kolumny: Obserwowane TAK/NIE
    labels = [['TP', 'FP'],  # Pred = 1
              ['FN', 'TN']]  # Pred = 0

    # Macierz z oznaczeniami
    labeled_cm = [[f"{cm[i][j]} ({labels[i][j]})" for j in range(2)] for i in range(2)]

    # Tworzymy ramkę wynikową
    cm_df = pd.DataFrame(
        labeled_cm,
        index=['Przewidywane TAK', 'Przewidywane NIE'],
        columns=['Obserwowane TAK', 'Obserwowane NIE']
    )

    # Dodajemy sumy brzegowe (jeśli wybrano)
    if margins:
        cm_numeric = pd.DataFrame(
            cm,
            index=['Przewidywane TAK', 'Przewidywane NIE'],
            columns=['Obserwowane TAK', 'Obserwowane NIE']
        )
        cm_numeric["Razem"] = cm_numeric.sum(axis=1)
        cm_numeric.loc["Razem"] = cm_numeric.sum(axis=0)

        # Łączenie liczby i oznaczenia
        for row in cm_df.index:
            cm_df.loc[row, "Razem"] = str(cm_numeric.loc[row, "Razem"])
        cm_df.loc["Razem"] = cm_numeric.loc["Razem"].astype(str)

    return cm_df






def predict_with_threshold(probabilities, threshold=0.5, true_labels=None):
    """
    Tworzy ramkę z prawdopodobieństwami, decyzją 0/1 i opcjonalnie stanem faktycznym.

    Parametry:
    - probabilities: lista lub tablica z przewidywanymi prawdopodobieństwami (klasa 1)
    - threshold: punkt odcięcia (default = 0.5)
    - true_labels: opcjonalna lista z rzeczywistymi etykietami (0/1)

    Zwraca:
    - Pandas DataFrame: kolumna 'Prawdopodobieństwo', 'Decyzja (1=TAK)', oraz opcjonalnie 'Rzeczywisty stan'
    """

    probabilities = np.array(probabilities)
    decisions = (probabilities >= threshold).astype(int)

    data = {
        "Prawdopodobieństwo": probabilities,
        "Decyzja (1=TAK)": decisions
    }

    if true_labels is not None:
        data = {
            "Rzeczywisty stan": np.array(true_labels),
            **data
        }

    return pd.DataFrame(data)


