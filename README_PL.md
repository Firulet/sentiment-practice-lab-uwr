# Sentiment & Emotion Analysis - Interactive Laboratory (wersja 2)

Nowa, niezależna aplikacja do przedmiotu **Introduction to Data Science and Society**.

## Co zmieniono

- Więcej eksperymentów na prawdziwych danych i mniej quizów; dotychczasowe pytania nie są automatycznie wyświetlane.
- Słownik, wieloetykietowe emocje, klucz kodowy, dwaj niezależni anotatorzy.
- Porównywanie etykiet GoEmotions po filtrowaniu danych i zmienianiu prób.
- TF-IDF (1–2-gramy) + regresja logistyczna one-vs-rest (8 klas): wyjaśnione prostym językiem.
- Prawdziwa ewaluacja modelu na niewidzianej części testowej GoEmotions: próg klasyfikacji, TP/FP/FN/TN, precision/recall/F1 i zmiana wypowiedzi.
- Osobny symulator ręcznej macierzy pomyłek, jednoznacznie oznaczony jako hipotetyczny.
- Sześć klasycznych zasad Cialdiniego oraz samodzielne kodowanie rzeczywistych wypowiedzi z PersuasionForGood.
- Końcowy projekt badawczy wykorzystujący prawdziwe etykiety GoEmotions, nie wymyśloną bazę.
- PL / EN i wygląd oparty na motywach Streamlit, bez kont studentów, zapisu odpowiedzi czy zewnętrznego API AI do ich tekstów.

## Ważne rozróżnienia badawcze

1. GoEmotions (Google Research): anotowane komentarze Reddita. Etykiety są ludzkimi interpretacjami, nie „prawdziwymi uczuciami”.
2. PersuasionForGood (Wang i in. 2019): rzeczywiste dialogi w zadaniu o darowiznach. W aplikacji student oznacza własne hipotezy kodowe według zasad Cialdiniego. Nie ma oficjalnego gold-standardu sześciu zasad w tym korpusie.
3. Sześć klasycznych zasad Cialdiniego to nie to samo co pięć kategorii perswazji z prezentacji o powodzi; autor obecnie wymienia też siódmą zasadę *unity*.
4. Model w aplikacji to klasyfikator TF-IDF + regresja logistyczna, NIE BERT. Wyniki TextBlob/BERT z wykładu kryzysowego to wyniki przytoczone, nie przeliczone w tej aplikacji.
5. GoEmotions i PersuasionForGood pobierane są przez aplikację z publicznych źródeł. Bez internetu interfejs jawnie zgłasza niedostępność danych i nie generuje fikcyjnych „wyników”.

## Bezpieczna aktualizacja istniejącego repozytorium GitHub

**Repozytorium:** `Firulet/sentiment-practice-lab-uwr`.

1. Na GitHubie pobierz kopię obecnego repozytorium: **Code > Download ZIP**.
2. W istniejącym repozytorium podmień w katalogu głównym pliki `app.py`, `content.py`, `requirements.txt`, `README_PL.md`; **dodaj** `experiments.py`.
3. Folder `.streamlit` zawiera `config.toml`; zachowaj jego lokalizację jako `.streamlit/config.toml`.
4. Zatwierdź zmiany przez **Commit changes**. Nie trzeba tworzyć nowej aplikacji: istniejący Streamlit automatycznie pobierze nowy commit.
5. Jeśli aplikacja nie wystartuje, otwórz **Manage app > Logs**. Można bezpiecznie wrócić do poprzedniego commita (lub przywrócić kopię).

Pliki PDF `Sentiment_Analysis_Lecture_PL_v2.pdf` i `Sentiment_Analysis_Lecture_EN_v2.pdf` są wykładami osobnymi; nie musisz wgrywać ich do repozytorium aplikacji.

## Test po wdrożeniu

- Przełącz Polski / English.
- Sprawdź stronę „Metody wyjaśnione”, Jaccard oraz macierz pomyłek.
- Włącz GoEmotions: sprawdź, czy pojawia się liczba komentarzy testowych.
- Wytrenuj model i zmień próg. Wartości TP, FP, FN, TN powinny się zmieniać po zmianie progu.
- W Perswazji poczekaj na pobranie publicznych dialogów; jeśli nie ma dostępu do GitHub raw, sprawdź łączność i logi.
- W ustawieniach aplikacji Streamlit można przełączyć wygląd na ciemny i upewnić się, że tekst jest czytelny.

## Wymagania i prywatność

Python, Streamlit, pandas, numpy, requests, scikit-learn, altair. Biblioteki są wymienione w `requirements.txt`. Nie ma logowania ani trwałego zapisu pracy studentów. Dane źródłowe zawierają wypowiedzi osób trzecich, dlatego nie należy dopisywać prywatnych wiadomości studentów do publicznego repozytorium.

## Źródła

- https://github.com/google-research/google-research/tree/master/goemotions
- https://convokit.infosci.cornell.edu/datasets.html
- https://github.com/persuasion-nlp/persuasion-strategies
- https://cialdini.com/principles
