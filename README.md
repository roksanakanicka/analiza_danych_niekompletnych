# Przewidywanie cen wynajmu mieszkań w Poznaniu

Projekt zajmuje się analizą i regresją cen wynajmu mieszkań na podstawie danych z rynku poznańskiego na cel zajęć Analiza Danych Niekompletnych 2025/2026.

**Analiza Danych Niekompletnych 2025/2026**

---

## Autorzy

- Maja Sklepowicz *(MajaSklp na Kaggle)*
- Roksana Kanicka *(roksanakanicka na Kaggle)*

**Wynik na Kaggle:** 85435.70 - 12. miejsce

---

## 1. Wstęp

Projekt koncentruje się na predykcji cen wynajmu mieszkań w Poznaniu na podstawie zbioru danych zawierającego parametry techniczne ofert oraz ich tytuły. Zastosowane podejście opiera się na czyszczeniu danych, różnorodnej imputacji braków oraz wykorzystaniu modeli Random Forest i XGBoost.

Dzięki optymalizacji procesu imputacji danych (w tym ekstrakcji informacji z tytułów ogłoszeń za pomocą wyrażeń regularnych) oraz zastosowaniu uśrednionej predykcji obu modeli, udało się uzyskać wynik **85435.70**, zajmując **12. miejsce** w konkursie na Kaggle.

W trakcie pracy korzystano z pomocy LLM-ów do analizy przykładów rozwiązań, poprawy struktury kodu oraz weryfikacji poprawności uzyskanych wyników.

---

## 2. Metodyka

### 2.1 Preprocessing i czyszczenie danych

Pracę rozpoczęto od zapewnienia pełnej powtarzalności wyników poprzez ustawienie `random_state=42` dla wszystkich algorytmów oraz bibliotek. Następnie przeprowadzono analizę eksploracyjną zbioru danych metodami `describe()`, `isnull()` oraz wizualizację braków za pomocą biblioteki `missingno`.

Zidentyfikowano cztery główne kategorie danych z brakami:
- puste pola w kolumnach logicznych (True/False),
- wartości 0 w czynszu i kaucji,
- luki w nazwach dzielnic,
- ujemne lub brakujące wartości metrażu i liczby pokoi.

Wszystkie wartości kodowane jako `-999`, `-9` oraz puste ciągi znaków zostały zamienione na `NaN` dla spójności. W przypadku kolumn `flat_rent` i `flat_deposit` dodatkowo zamieniono wartości zerowe na `NaN`, ponieważ czynsz i kaucja równe zero nie mają sensu ekonomicznego i najprawdopodobniej oznaczają brak informacji.

Dla cech typu boolean (np. `flat_balcony`, `flat_garage`, `flat_internet`) przyjęto założenie, że brak zaznaczenia opcji oznacza jej brak w rzeczywistości — wynajmujący zazwyczaj podkreślają wszystkie atuty mieszkania. Dlatego wszystkie puste wartości w tych kolumnach wypełniono wartością `False`.

---

### 2.2 Imputacja oparta na tekście (NLP)

Najważniejszym krokiem była **ekstrakcja informacji z tytułów ogłoszeń** przy użyciu wyrażeń regularnych (Regex).

Stworzono funkcję `imputacja_pokoje()`, która "czytała" tytuły ogłoszeń w poszukiwaniu wzorców, np.:
- `2-pok`, `dwupok`, `dwa pok` → 2 pokoje
- `kawalerk`, `studi`, `apartament` → 1 pokój
- `3-pok`, `trzy pok` → 3 pokoje

Ta metoda pozwoliła odzyskać informację o liczbie pokoi dla około **90% mieszkań** z brakami w kolumnie `flat_rooms`. Wartości powyżej lub równe 7 korygowano do 1, uznając je za błąd wprowadzenia danych.

Analogicznie stworzono funkcję `uzupelnij_area_i_quarter()` do ekstrakcji:
- **Metrażu** z wzorców typu `"45 m2"`, `"60.5 mkw"`, `"38,5 m²"` z walidacją zakresu 10–200 m²,
- **Nazw dzielnic** poprzez wyszukiwanie oficjalnych nazw osiedli Poznania w tytułach.

Dzielnice sortowano malejąco po długości nazwy, aby uniknąć fałszywych dopasowań (np. `"Nowe Miasto"` przed `"Miasto"`).

---

### 2.3 Imputacja numeryczna

Dla pozostałych braków numerycznych zastosowano **IterativeImputer**, który estymuje wartości na podstawie wzajemnych zależności między zmiennymi:
- `flat_area` (metraż)
- `flat_rooms` (liczba pokoi)
- `quarter_copy` (dzielnica zakodowana numerycznie)
- `building_floor_num` (liczba pięter w budynku)

Przed użyciem IterativeImputer, nazwy dzielnic zakodowano numerycznie za pomocą `LabelEncoder`, grupując rzadkie dzielnice (< 10 wystąpień) w kategorię `"Inne"`, aby uniknąć przeuczenia modelu na małych grupach. Po zakończeniu imputacji wartości dyskretne zostały zaokrąglone do liczb całkowitych.

---

### 2.4 Imputacja czynszu i kaucji

Dla zmiennych `flat_rent` i `flat_deposit` zastosowano **imputację grupową** opartą na medianie według liczby pokoi:

```python
rent_map    = train.groupby('flat_rooms')['flat_rent'].median()
deposit_map = train.groupby('flat_rooms')['flat_deposit'].median()
```

Czynsz i kaucja silnie korelują z liczbą pokoi. Użycie mediany zamiast średniej jest bardziej odporne na wartości odstające. Dla mieszkań, których liczby pokoi nie udało się ustalić, użyto globalnej mediany całego zbioru jako fallback.

Po zakończeniu imputacji zakodowane numerycznie nazwy dzielnic zostały przekształcone z powrotem na oryginalne nazwy przy użyciu `inverse_transform`.

---

### 2.5 Feature engineering

Na końcu etapu przetwarzania dodano trzy nowe cechy binarne oparte na słowach kluczowych w tytułach:

| Cecha | Opis |
|---|---|
| `is_prestige` | Nazwy prestiżowych inwestycji (City Park, Stary Browar, Ostrów Tumski itp.) |
| `is_premium` | Słowa wskazujące na wysoki standard (apartament, luksus, klimatyzacja, taras, sauna itp.) |
| `is_budget` | Słowa wskazujące na tańsze oferty (tani, okazja, student, pokój itp.) |

Model początkowo słabo radził sobie z drogimi mieszkaniami — te cechy pomogły algorytmowi lepiej rozpoznać segment cenowy oferty, co znacząco wpłynęło na jakość predykcji.

---

### 2.6 Modelowanie

Usunięto cechy niemające wpływu na cenę: `id`, `ad_title`, `date_activ`, `date_modif`, `date_expire`, `price` oraz `quarter` (zastąpiona przez `quarter_copy`).

Zastosowano podejście **ensemble learning**, łącząc dwa algorytmy:

#### Random Forest Regressor
- `n_estimators=100`
- Świetnie radzi sobie z nieliniowymi zależnościami
- Odporny na przeuczenie dzięki agregacji wielu drzew decyzyjnych

#### XGBoost Regressor
- `n_estimators=1500`, `learning_rate=0.015`, `max_depth=7`
- Zmienna docelowa poddana **transformacji logarytmicznej** (`log1p`)
- Rozkład cen był prawoskośny — transformacja log znormalizowała rozkład, zmniejszając wpływ wartości odstających i poprawiając dokładność predykcji

Ostateczna predykcja stanowi **średnią arytmetyczną** wyników obu modeli:

```python
final_pred = (rf_pred + xgb_pred) / 2
```

Uśrednianie predykcji z różnych algorytmów (ensemble) zazwyczaj daje lepsze wyniki niż pojedynczy model, ponieważ błędy poszczególnych modeli częściowo się znoszą.

---

## 3. Wyniki

### Walidacja lokalna

Skuteczność modelu zweryfikowano za pomocą `train_test_split` (80% train, 20% walidacja):

| Metryka | Wartość |
|---|---|
| MAE | 196.69 |
| RMSE | 287.70 |
| R² Score | 0.73 |

Stworzono wykres scatter plot porównujący prawdziwe ceny z predykcjami. Analiza wykazała, że model najlepiej radzi sobie z mieszkaniami w średnim przedziale cenowym, natomiast dla bardzo drogich ofert występuje większy rozrzut predykcji związany z niedoszacowaniem.

### Wynik na Kaggle

**85435.70 — 12. miejsce**

Kluczowe etapy poprawy wyniku:
1. Etapowe dodawanie ekstrakcji cech z tytułu ogłoszenia
2. Zastąpienie prostej imputacji przez IterativeImputer
3. Feature engineering: dodanie cech `is_prestige`, `is_premium`, `is_budget`
4. Zmiana modelu pojedynczego na ensemble (RF + XGBoost)

---

## 4. Podsumowanie

Projekt wykazał, że kluczowym etapem nie jest sam wybór modelu, lecz **jakość przygotowania danych**. W przypadku danych z portali ogłoszeniowych, gdzie użytkownicy wprowadzają informacje w sposób niespójny, imputacja oparta na analizie tekstu (NLP) bardzo skutecznie polepszyła wyniki modelu.

Projekt pozwolił na praktyczne zastosowanie tradycyjnych metod imputacji, technik NLP opartych na Regex oraz zaawansowanych modeli uczenia maszynowego (XGBoost, Random Forest) w podejściu ensemble.

LLM-y (ChatGPT, Gemini) znacznie przyspieszyły pracę — szczególnie w znajdowaniu błędów, dobieraniu hiperparametrów do XGBoost oraz generowaniu wzorców Regex. Jednocześnie zdarzało im się proponować zbyt skomplikowane rozwiązania lub sugestie, które nie działały poprawnie — co uwrażliwiło nas na konieczność krytycznej weryfikacji generowanego kodu.

## Jak uruchomić:
1. Stworzenie środowiska wirtualnego: `python -m venv venv`
2. Aktywacja: `source venv/Scripts/activate` (Windows) lub `source venv/bin/activate` (Mac/Linux)
3. Zainstalowanie bibliotek: `pip install -r requirements.txt`
4. Uruchomienie skryptu: `python projekcik.py`
