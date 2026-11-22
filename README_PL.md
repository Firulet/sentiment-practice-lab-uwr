# Sentiment & Emotion Analysis — osobna aplikacja Streamlit

Kurs: **Introduction to Data Science and Society**.

## Co jest w środku?

- `app.py` i `content.py` — kod nowego laboratorium (PL / EN).
- `.streamlit/config.toml` — jasny motyw początkowy; teksty i wykresy respektują także systemowy tryb ciemny Streamlit (przełączany przez użytkownika w menu aplikacji).
- `requirements.txt` — zależności.
- **24 pytania z objaśnieniami**, 10 scenariuszy interpretacyjnych, symulator słownika, wieloetykietowe emocje, rozkład etykiet, macierz pomyłek, interaktywna ewaluacja i opcjonalny klasyfikator TF-IDF + regresja logistyczna.
- Brak zapisu odpowiedzi, numerów indeksów i połączenia Google Sheets.

## Publikacja jako osobna aplikacja

1. Nie ruszaj `Firulet/data-society-research-academy` ani `sna-practice-lab-uwr`.
2. Na GitHub utwórz **nowe repozytorium** (np. `sentiment-practice-lab-uwr`).
3. Wgraj **pliki z wnętrza katalogu**, tak by `app.py`, `content.py` i `requirements.txt` były w katalogu głównym. Wgraj też ukryty katalog `.streamlit` (jeśli GitHub nie pozwala przeciągnąć go, utwórz `.streamlit/config.toml` ręcznie).
4. Otwórz https://share.streamlit.io/ , kliknij Create app, wskaż repo, gałąź `main`, ścieżkę `app.py`.
5. Wybierz np. `sentiment-practice-lab-uwr` jako własny adres, jeśli wolny, i kliknij Deploy.
6. Po wdrożeniu sprawdź `English / Polski`, 24 quizy, symulator, pobranie GoEmotions i trening modelu. Zmiana motywu: menu Streamlit (trzy kropki) → Settings → Theme (Light / Dark).

## Ważne granice i dane

- *GoEmotions* to rzeczywiste anglojęzyczne komentarze Reddita z ludzkimi anotacjami 27 emocji + neutral. Aplikacja pobiera **oficjalne pliki test/train** z `https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/`. Jest wymagany internet podczas pierwszego pobrania.
- Gdy źródło nie jest dostępne, działa tryb demonstracyjny z wyraźnie oznaczonymi **autorskimi zdaniami**, bez udawania prawdziwego zbioru. Funkcje modelowe są wtedy wyłączone.
- Model dydaktyczny **nie jest BERT-em**. Trenuje TF-IDF + 8 niezależnych klasyfikatorów logistycznych (one-vs-rest) na próbie 7 500 komentarzy. Wyniki są przybliżonymi score'ami, a nie oceną rzeczywistych uczuć.
- Słownik dotyczy angielskich słów (nie polskich). Język interfejsu można zmieniać niezależnie od języka danych.
- Wybrane scenariusze komunikacji kryzysowej są **fikcyjne**. Rozkłady TextBlob i BERT w module są wyłącznie cytowanymi wynikami z dostarczonej prezentacji, nie estymacją aplikacji.
- Nie wklejaj do publicznej aplikacji prywatnych wiadomości uczestników badań.

## Test na własnym komputerze

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Źródła: https://github.com/google-research/google-research/tree/master/goemotions ; https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
