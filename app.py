"""Bilingual, standalone educational sentiment-analysis laboratory.
No accounts, no collection, no persistent submission, no private data.
Real GoEmotions data are fetched on demand from the official Google Research repo.
"""
import io
import re
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from content import EMOTIONS, EMO_PL, POS, NEG, LEX_POS, LEX_NEG, QUESTIONS, SCENARIOS

st.set_page_config(page_title="Sentiment & Emotion Lab | Data Science and Society",page_icon="💬",layout="wide",initial_sidebar_state="expanded")
st.markdown('''<style>
.block-container{max-width:1160px;padding-top:1.5rem}
.course-box{border:1px solid var(--secondary-background-color);border-left:4px solid #17A398;border-radius:12px;padding:16px 20px;margin:14px 0;background:var(--secondary-background-color);color:var(--text-color)}
.small-note{opacity:.8;font-size:.9rem}
.stButton>button{border-radius:10px}
</style>''',unsafe_allow_html=True)

st.sidebar.markdown("### 🌍 Language / Język")
lang=st.sidebar.segmented_control("Language",options=["English","Polski"],default="English",label_visibility="collapsed")
pl=lang=="Polski"
def tr(en,po):return po if pl else en
def emo(x):return EMO_PL[EMOTIONS.index(x)] if pl and x in EMOTIONS else x

def note(en,po):st.markdown(f'<div class="course-box">{tr(en,po)}</div>',unsafe_allow_html=True)

st.title(tr("💬 Sentiment & Emotion Analysis","💬 Analiza sentymentu i emocji"))
st.caption("Introduction to Data Science and Society · University teaching laboratory · English / Polski")
note("**From word lists to social meaning.** Compare lexicon scores, real human annotations and a simple trained classifier. This lab is not BERT and does not claim to read people's feelings.","**Od listy słów do znaczenia społecznego.** Porównaj wyniki słownika, rzeczywiste anotacje i prosty klasyfikator uczony. To laboratorium nie korzysta z BERT-a i nie odczytuje rzeczywistych uczuć ludzi.")

GO_BASE="https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
@st.cache_data(ttl=24*3600,show_spinner=False)
def fetch_split(name):
    url=GO_BASE+name+".tsv"
    response=requests.get(url,timeout=18)
    response.raise_for_status()
    rows=[]
    for line in response.text.splitlines():
        pieces=line.split("\t")
        if len(pieces)<3:continue
        try:
            ids=[int(v) for v in pieces[1].split(",")]
            labels=[EMOTIONS[i] for i in ids if 0<=i<len(EMOTIONS)]
        except (ValueError,IndexError):continue
        if labels:rows.append({"text":pieces[0],"labels":labels,"id":pieces[2]})
    if not rows:raise ValueError("The source returned no usable rows")
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner=False)
def train_classifier():
    df=fetch_split("train")
    # A compact, reproducible teaching baseline; NOT a transformer.
    training=df.sample(n=min(7500,len(df)),random_state=23)
    selected=["joy","anger","sadness","fear","gratitude","surprise","optimism","neutral"]
    Y=np.asarray([[int(e in labels) for e in selected] for labels in training.labels],dtype=int)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=11000,stop_words="english")
    X=vec.fit_transform(training.text)
    clf=OneVsRestClassifier(LogisticRegression(max_iter=220,class_weight="balanced",random_state=23))
    clf.fit(X,Y)
    return vec,clf,selected

@st.cache_data(show_spinner=False)
def load_demo():
    return pd.DataFrame([{"text":e,"labels":["neutral"],"id":f"illustration_{i}"} for i,(e,_,_) in enumerate(SCENARIOS)])

try:
    test=fetch_split("test")
    real=True
except Exception:
    test=load_demo()
    real=False
if real:
    st.sidebar.success(tr(f"Real GoEmotions test split: {len(test):,} comments",f"Rzeczywisty zbiór testowy GoEmotions: {len(test):,} komentarzy"))
else:
    st.sidebar.warning(tr("Dataset server unavailable: clearly labelled illustrative examples only. No original labels are shown.","Serwer danych niedostępny: pokazujemy jedynie wyraźnie oznaczone przykłady autorskie. Nie pokazujemy etykiet źródłowych."))

nav=[("start",tr("Start: data and concepts","Start: dane i pojęcia")),("lexicon",tr("Word-list simulator","Symulator słownika")),("data",tr("Explore real GoEmotions","Poznaj rzeczywiste GoEmotions")),("multi",tr("Multiple emotions","Wiele emocji")),("model",tr("Train and compare models","Model i porównanie")),("metrics",tr("Evaluation laboratory","Laboratorium ewaluacji")),("cases",tr("Social research scenarios","Scenariusze socjologiczne")),("quiz",tr("24 questions and answers","24 pytania i odpowiedzi")),("reflection",tr("Final research challenge","Końcowe zadanie badawcze"))]
st.sidebar.markdown("### " + tr("Choose an activity","Wybierz ćwiczenie"))
section=st.sidebar.radio("Activities",[x[0] for x in nav],format_func=lambda x:dict(nav)[x],label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(tr("No student names, accounts, uploads or answer submissions. All answers stay in the current Streamlit session.","Bez nazwisk, kont, wgrywania i wysyłania odpowiedzi. Odpowiedzi są tylko w bieżącej sesji Streamlit."))


def lex(text,negation=False):
    tokens=re.findall(r"[a-z]+",text.lower())
    pos_score=neg_score=0
    details=[]
    for i,word in enumerate(tokens):
        val=LEX_POS.get(word,0)-LEX_NEG.get(word,0)
        before=tokens[max(0,i-2):i]
        if negation and any(x in {"not","never","no","don't","isn't","wasn't","cannot"} for x in before):val=-val
        if val:
            details.append((word,val))
            if val>0:pos_score+=val
            else:neg_score-=val
    return pos_score-neg_score,details,pos_score,neg_score

def sentiment_label(score):
    return tr("Positive","Pozytywny") if score>0 else (tr("Negative","Negatywny") if score<0 else tr("Undetermined","Nierozstrzygnięty"))

def quiz_widget(prefix,items):
    for j,q in items:
        section_q,qpl,qen,optpl,opten,answer,whypl,whyen=q
        st.markdown(f"**{j+1}. {qpl if pl else qen}**")
        options=optpl if pl else opten
        key=f"{prefix}_{j}"
        selected=st.radio("Select",[0,1,2],index=None,format_func=lambda i:options[i],key=key,label_visibility="collapsed")
        if st.button(tr("Check and explain","Sprawdź i wyjaśnij"),key=f"check_{key}"):
            if selected is None:st.info(tr("Choose an answer first.","Najpierw wybierz odpowiedź."))
            else:
                if selected==answer:st.success(tr("Correct!","Poprawnie!"))
                else:st.error(tr("Not quite. Correct answer:","Nie całkiem. Poprawna odpowiedź:")+" "+options[answer])
                st.info(whypl if pl else whyen)
        st.divider()

if section=="start":
    st.header(tr("How can computational methods interpret social text?","Jak metody obliczeniowe analizują tekst społeczny?"))
    a,b,c=st.columns(3)
    a.metric(tr("Original UCI reviews","Dawne recenzje UCI"),"3,000",tr("positive/negative","pozytywne/negatywne"))
    b.metric(tr("GoEmotions (entire corpus)","GoEmotions (cały korpus)"),"58,009",tr("Reddit comments","komentarzy Reddit"))
    c.metric(tr("GoEmotions labels","Etykiety GoEmotions"),"27+1",tr("emotions + neutral","emocje + neutralna"))
    st.subheader(tr("Three different research tasks","Trzy różne zadania badawcze"))
    st.dataframe(pd.DataFrame({tr("Task","Zadanie"):[tr("Sentiment polarity","Polaryzacja sentymentu"),tr("Emotion recognition","Rozpoznawanie emocji"),tr("Stance detection","Wykrywanie stanowiska")],tr("Example","Przykład"):[tr("positive / neutral / negative","pozytywny / neutralny / negatywny"),tr("fear + gratitude","strach + wdzięczność"),tr("in favour of / against a proposal","za / przeciw propozycji")]}),hide_index=True,use_container_width=True)
    note("**Dataset caution:** GoEmotions uses human annotations of Reddit comments, not measurements of inner emotions. The UCI review dataset deliberately contains positive/negative sentences and no neutral class.","**Uwaga o danych:** GoEmotions to anotacje komentarzy Reddita, a nie pomiary wewnętrznych emocji. Zbiór UCI zawiera celowo dobrane recenzje pozytywne i negatywne, bez klasy neutralnej.")
    st.markdown(tr("**Try:** Is 'We need volunteers immediately' neutral in polarity but urgent in communicative function?","**Zastanów się:** Czy 'Potrzebujemy natychmiast ochotników' może być neutralne w polaryzacji, ale pilne komunikacyjnie?"))
    quiz_widget("start",[(i,q) for i,q in enumerate(QUESTIONS) if q[0]=="basics"])

elif section=="lexicon":
    st.header(tr("Word-list sentiment simulator","Symulator analizy słownikowej"))
    st.write(tr("A transparent teaching algorithm: split the input into English words, add weights for positive words and subtract weights for negative words.","Przejrzysty algorytm dydaktyczny: dzielimy tekst na angielskie słowa, dodajemy wagi słów pozytywnych i odejmujemy wagi negatywnych."))
    example=st.selectbox(tr("Choose a scenario","Wybierz scenariusz"),list(range(len(SCENARIOS))),format_func=lambda i:SCENARIOS[i][1] if pl else SCENARIOS[i][0])
    user_text=st.text_area(tr("English text to analyse","Angielski tekst do analizy"),value=SCENARIOS[example][0],height=110)
    rule=st.checkbox(tr("Use a simple negation reversal for the preceding two words","Odwracaj prostą negację w dwóch poprzednich słowach"),value=False)
    score,terms,p,n=lex(user_text,rule)
    c1,c2,c3=st.columns(3)
    c1.metric(tr("Weighted positive terms","Waga słów pozytywnych"),p)
    c2.metric(tr("Weighted negative terms","Waga słów negatywnych"),n)
    c3.metric(tr("Net score","Wynik netto"),score)
    st.write(tr("**Classification:** ","**Klasyfikacja:** ")+sentiment_label(score))
    if terms:
        st.dataframe(pd.DataFrame({tr("Matched word","Znalezione słowo"):[x[0] for x in terms],tr("Contribution","Wkład"):[x[1] for x in terms]}),hide_index=True)
    else:st.info(tr("No words were matched. This is not proof of emotional neutrality.","Nie znaleziono słów ze słownika. To nie dowodzi neutralności emocjonalnej."))
    st.latex(r"S = \sum_{w\in text} positive(w) - \sum_{w\in text} negative(w)")
    st.bar_chart(pd.DataFrame({tr("Score","Wynik"):[p,n]},index=[tr("Positive","Pozytywne"),tr("Negative","Negatywne")]))
    st.warning(tr("This is an authored mini-lexicon, NOT TextBlob, BERT or a validated sentiment tool. English word matching only.","To autorski minisłownik, NIE TextBlob, BERT ani zwalidowane narzędzie. Rozpoznaje tylko angielskie słowa."))
    quiz_widget("lex",[(i,q) for i,q in enumerate(QUESTIONS) if q[0]=="lexicon"])

elif section=="data":
    st.header(tr("Explore the GoEmotions dataset","Eksploracja zbioru GoEmotions"))
    if real:
        st.write(tr("These examples are real test-set comments with human annotations. A comment may have several labels.","To rzeczywiste komentarze ze zbioru testowego z anotacjami ludzkimi. Komentarz może mieć kilka etykiet."))
        order=st.slider(tr("Comment number","Numer komentarza"),0,min(500,len(test)-1),12)
        row=test.iloc[order]
        st.markdown(f'> {row["text"]}')
        if st.button(tr("Reveal reference annotation","Pokaż anotację źródłową")):
            st.success(", ".join(emo(v) for v in row["labels"]))
        st.subheader(tr("How many labels can a comment have?","Ile etykiet może mieć komentarz?"))
        counts=test.labels.map(len).value_counts().sort_index()
        st.bar_chart(counts.rename(tr("Number of comments","Liczba komentarzy")))
        st.subheader(tr("Most frequent emotion labels in the test set","Najczęstsze etykiety emocji w zbiorze testowym"))
        counter=Counter(e for labels in test.labels for e in labels)
        chart=pd.DataFrame({tr("Emotion","Emocja"):[emo(x) for x,_ in counter.most_common(15)],tr("Occurrences","Wystąpienia"):[v for _,v in counter.most_common(15)]}).set_index(tr("Emotion","Emocja"))
        st.bar_chart(chart)
        st.caption(tr("Counts are label occurrences, not mutually exclusive percentages. The complete 58k-comment corpus is described by the source; the app loads the official test partition.","Liczby to wystąpienia etykiet, a nie wykluczające się procenty. Źródło opisuje cały korpus 58 tys., aplikacja ładuje oficjalną część testową."))
    else:
        st.warning(tr("The official data could not be downloaded. Return later for real examples. The remaining simulations still work.","Nie udało się pobrać danych oficjalnych. Wróć później po prawdziwe przykłady. Pozostałe symulacje działają."))
    quiz_widget("data",[(i,q) for i,q in enumerate(QUESTIONS) if q[0]=="emotions"])

elif section=="multi":
    st.header(tr("Multiple emotions and disagreement","Wiele emocji i niezgodność ocen"))
    st.write(tr("A human can perceive more than one emotion in the same sentence. Select the labels you see and compare with another hypothetical rater.","Człowiek może rozpoznać więcej niż jedną emocję w zdaniu. Wybierz dostrzegane emocje i porównaj z drugim hipotetycznym anotatorem."))
    ex=st.selectbox(tr("Example","Przykład"),list(range(len(SCENARIOS))),format_func=lambda i:SCENARIOS[i][1] if pl else SCENARIOS[i][0],key="multiscenario")
    st.write(SCENARIOS[ex][0])
    selected=st.multiselect(tr("Your emotion labels","Twoje etykiety emocji"),EMOTIONS,format_func=emo)
    st.metric(tr("Number of selected labels","Liczba wybranych etykiet"),len(selected))
    st.info(tr("There is no automatic 'correct' answer for these authored examples. In scientific research, annotators need a codebook and agreement rules.","Dla tych autorskich przykładów nie ma automatycznie jedynej poprawnej odpowiedzi. W badaniach potrzebny jest klucz kodowy i reguły zgodności."))
    a=st.multiselect(tr("Rater A","Anotator A"),EMOTIONS,default=["fear","nervousness"],format_func=emo,key="rA")
    b=st.multiselect(tr("Rater B","Anotator B"),EMOTIONS,default=["fear","surprise"],format_func=emo,key="rB")
    union=set(a)|set(b);jaccard=len(set(a)&set(b))/len(union) if union else 1.0
    st.metric(tr("Jaccard agreement (illustration)","Zgodność Jaccarda (przykład)"),f"{jaccard:.2f}")
    st.latex(r"J(A,B)=\frac{|A\cap B|}{|A\cup B|}")
    st.caption(tr("This is a simple set-overlap illustration, not a substitute for a validated inter-rater reliability study.","To przykład nakładania się zbiorów, a nie pełna analiza rzetelności między anotatorami."))

elif section=="model":
    st.header(tr("From keywords to a trained classifier","Od słów kluczowych do modelu uczonego"))
    note("**Model:** TF-IDF (1–2-grams) + one-vs-rest logistic regression, trained on a reproducible sample of official GoEmotions training rows. This is NOT BERT. It is a teaching baseline for 8 selected labels.","**Model:** TF-IDF (1–2-gramy) + regresja logistyczna one-vs-rest, uczona na powtarzalnej próbie oficjalnych danych treningowych GoEmotions. To NIE BERT. Jest to dydaktyczny model bazowy dla 8 wybranych etykiet.")
    if real:
        st.write(tr("Model training is optional and can take a short moment. No student text is sent to any external AI API.","Uczenie modelu jest opcjonalne. Tekst studentów nie jest wysyłany do zewnętrznego API AI."))
        if st.button(tr("Train/load the classroom model","Naucz/wczytaj model dydaktyczny"),type="primary"):
            with st.spinner(tr("Training baseline...","Uczenie modelu...")):
                try:train_classifier();st.session_state["model_ready"]=True
                except Exception as ex:st.error(tr("Could not download/train the model: ","Nie udało się pobrać/wyuczyć modelu: ")+str(ex)[:120])
        if st.session_state.get("model_ready"):
            vec,clf,labels=train_classifier()
            example=st.selectbox(tr("Try an example","Wybierz przykład"),[e[0] for e in SCENARIOS],key="model_example")
            txt=st.text_area(tr("Your own English message","Własna angielska wiadomość"),value=example,key="model_text")
            scores=clf.predict_proba(vec.transform([txt]))[0]
            indexed=sorted(zip(labels,scores),key=lambda a:-a[1])
            st.bar_chart(pd.DataFrame({tr("Score","Wynik"):[v for _,v in indexed]},index=[emo(e) for e,_ in indexed]))
            lex_score,*_=lex(txt)
            st.write(tr("Lexicon result:","Wynik słownika:"),sentiment_label(lex_score),"·",tr("highest model score:","najwyższy wynik modelu:"),emo(indexed[0][0]))
            st.caption(tr("Scores are separate binary classifier outputs; they need not sum to 1 and are not automatically calibrated.","Wyniki pochodzą z osobnych klasyfikatorów binarnych; nie muszą sumować się do 1 i nie są automatycznie skalibrowane."))
    else:st.warning(tr("The external data cannot currently be loaded, so training is disabled. No fake model predictions are generated.","Obecnie nie można pobrać danych, dlatego uczenie jest wyłączone. Nie generujemy fikcyjnych predykcji modelu."))
    quiz_widget("model",[(i,q) for i,q in enumerate(QUESTIONS) if q[0]=="models"])

elif section=="metrics":
    st.header(tr("Interactive precision, recall and F1","Interaktywne precision, recall i F1"))
    st.write(tr("Adjust a confusion matrix for one chosen emotion. The values here are a hypothetical simulation, not measured model results.","Zmieniaj macierz pomyłek dla jednej wybranej emocji. To symulacja hipotetyczna, a nie zmierzone wyniki modelu."))
    c1,c2,c3,c4=st.columns(4)
    tp=c1.slider("TP",0,100,32);fp=c2.slider("FP",0,100,8)
    fn=c3.slider("FN",0,100,12);tn=c4.slider("TN",0,100,48)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    a,b,c=st.columns(3);a.metric("Precision",f"{precision:.3f}");b.metric("Recall",f"{recall:.3f}");c.metric("F1",f"{f1:.3f}")
    st.dataframe(pd.DataFrame([[tp,fn],[fp,tn]],index=[tr("Actual positive","Rzeczywiście pozytywne"),tr("Actual negative","Rzeczywiście negatywne")],columns=[tr("Predicted positive","Przewidziane pozytywne"),tr("Predicted negative","Przewidziane negatywne")]),use_container_width=True)
    st.latex(r"Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad F1=\frac{2PR}{P+R}")
    st.write(tr("Try increasing false positives (FP). What happens to precision? Now increase false negatives (FN). What happens to recall?","Zwiększ fałszywie pozytywne (FP). Co dzieje się z precision? Potem zwiększ fałszywie negatywne (FN). Co dzieje się z recall?"))
    quiz_widget("eval",[(i,q) for i,q in enumerate(QUESTIONS) if q[0]=="evaluation"])

elif section=="cases":
    st.header(tr("Social research scenarios and simulations","Scenariusze i symulacje socjologiczne"))
    st.write(tr("The following texts are **authored teaching scenarios**, not real crisis-group messages or GoEmotions examples.","Poniższe wypowiedzi to **autorskie scenariusze dydaktyczne**, a nie prawdziwe wiadomości z grupy kryzysowej ani przykłady GoEmotions."))
    for i,(en,po,typ) in enumerate(SCENARIOS):
        with st.expander(f"{i+1}. {po if pl else en}"):
            st.write(tr("What could a word list miss? Identify possible sentiment, emotion, communicative function, and missing context.","Co może przeoczyć słownik? Określ możliwy sentyment, emocje, funkcję wypowiedzi i brakujący kontekst."))
            result=st.text_area(tr("Your interpretation (practice only)","Twoja interpretacja (ćwiczenie)"),key=f"interp_{i}")
            if st.button(tr("Compare with methodological hints","Pokaż wskazówki metodologiczne"),key=f"hint_{i}"):
                hints={"mixed":tr("Positive and negative cues can coexist. A single polarity score compresses this complexity.","Sygnały pozytywne i negatywne mogą współwystępować. Jeden wynik polaryzacji upraszcza tę złożoność."),"negation":tr("A lexicon that ignores negation may reverse the intended evaluation.","Słownik ignorujący negację może odwrócić znaczenie oceny."),"sarcasm":tr("Literal positive words can express criticism; audience context matters.","Dosłownie pozytywne słowa mogą wyrażać krytykę; ważny jest kontekst."),"crisis":tr("Urgency and perceived threat may not be explicit in lexical polarity.","Pilność i zagrożenie mogą nie wynikać bezpośrednio z polaryzacji słów.")}
                st.info(hints[typ])
    st.subheader(tr("Compare approaches from a crisis communication study","Porównanie podejść w badaniu komunikacji kryzysowej"))
    st.write(tr("The supplied course presentation reports that TextBlob and a BERT-based approach produced different distributions of messages from a Wrocław flood-threat WhatsApp group. These are **reported study results**, not outputs calculated by this app.","W załączonej prezentacji podano, że TextBlob i podejście oparte na BERT dały różne rozkłady wiadomości z grupy WhatsApp podczas zagrożenia powodziowego we Wrocławiu. Są to **wyniki opisanego badania**, a nie obliczenia tej aplikacji."))
    st.dataframe(pd.DataFrame({"TextBlob (3 labels)":["93.82% neutral","5.73% positive","0.45% negative"],"BERT (5 labels)":["38.8% very negative","21.1% very positive","18.4% neutral"]}),hide_index=True)
    st.caption(tr("The two approaches use different label sets. Do not treat this table as a matched accuracy evaluation. See the original course deck for the complete five-label BERT breakdown.","Metody używają różnych zbiorów etykiet. Tabela nie jest porównaniem dokładności modeli. Pełne pięć kategorii BERT znajduje się w prezentacji źródłowej."))

elif section=="quiz":
    st.header(tr("Concept questions – learn by explaining","Pytania pojęciowe – ucz się, wyjaśniając"))
    chosen=st.selectbox(tr("Concept area","Dział tematyczny"),["all","basics","lexicon","emotions","models","evaluation","ethics"],format_func=lambda x:{"all":tr("All 24 questions","Wszystkie 24 pytania"),"basics":tr("Basics","Podstawy"),"lexicon":tr("Lexicon","Słownik"),"emotions":tr("Emotions","Emocje"),"models":tr("Models","Modele"),"evaluation":tr("Evaluation","Ewaluacja"),"ethics":tr("Ethics","Etyka")}[x])
    subset=[(i,q) for i,q in enumerate(QUESTIONS) if chosen=="all" or q[0]==chosen]
    quiz_widget("allquiz",subset)

elif section=="reflection":
    st.header(tr("Your final research challenge","Końcowe wyzwanie badawcze"))
    st.write(tr("You are studying messages from students discussing AI in higher education. Design a scientifically responsible sentiment study.","Badasz wiadomości studentów dyskutujących o AI na uczelni. Zaprojektuj odpowiedzialne naukowo badanie sentymentu."))
    prompts=[( "Define the unit of analysis and sampling strategy.","Zdefiniuj jednostkę analizy i sposób doboru próby."),("Choose between polarity labels and multi-label emotions. Explain why.","Wybierz polaryzację lub wieloetykietowe emocje. Uzasadnij."),("How will you check disagreement among human annotators?","Jak sprawdzisz niezgodności między ludzkimi anotatorami?"),("How will you validate a model on unseen examples?","Jak zweryfikujesz model na niewidzianych przykładach?"),("What can you infer and what remains uncertain?","Co można wnioskować, a co pozostaje niepewne?"),("How will you protect participants and third parties?","Jak ochronisz uczestników i osoby trzecie?")]
    for i,(en,po) in enumerate(prompts):
        st.text_area(f"{i+1}. {po if pl else en}",key=f"research_{i}",height=75)
    st.info(tr("Practice-only text boxes: no saving or sending. You may copy your answers manually.","Pola służą wyłącznie do ćwiczeń: bez zapisu i wysyłania. Możesz ręcznie skopiować swoje odpowiedzi."))
    quiz_widget("ethics",[(i,q) for i,q in enumerate(QUESTIONS) if q[0]=="ethics"])

st.divider()
st.caption("Sources: Kotzias et al. (2015), UCI Sentiment Labelled Sentences · Demszky et al. (2020), GoEmotions (Google Research) · Course case study: Sentiment Analysis in Crisis Situations (provided lecture presentation). No persistent student-data collection.")
