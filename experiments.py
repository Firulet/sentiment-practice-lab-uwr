"""Hands-on social-science labs. No persistence or third-party student text transmission."""
from collections import Counter
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st

SIX = [
 ('reciprocity','Wzajemność','Reciprocity','Ktoś wcześniej nam pomógł; czujemy zobowiązanie do odwzajemnienia.','A previous favor creates a felt obligation to return help.','I helped you last week. Can you help me with this today?'),
 ('commitment','Zaangażowanie i konsekwencja','Commitment and consistency','Wcześniejsza deklaracja może skłaniać do spójnego działania.','Previous commitments can motivate consistent behavior.','You said you would volunteer. Can you confirm your shift?'),
 ('social_proof','Społeczny dowód słuszności','Social proof','Patrzymy, co robią podobni do nas ludzie.','Others’ actions, especially peers’, can guide decisions.','Most students in your year have already signed up.'),
 ('authority','Autorytet','Authority','Źródło przedstawia wiedzę lub wiarygodną ekspertyzę.','An expert or credible source supports a claim.','The public health agency recommends this action.'),
 ('liking','Lubienie i sympatia','Liking','Prośba może być skuteczniejsza, gdy lubimy nadawcę.','A request can be more persuasive when we like the source.','We know each other from class — could you help our team?'),
 ('scarcity','Niedostępność','Scarcity','Rzadkość lub ograniczony czas zwiększa postrzeganą wartość.','Limited availability or time can increase perceived value.','Only three seats remain and registration closes tonight.'),
]
@st.cache_data(ttl=24*3600,show_spinner=False)
def fetch_persuasion():
    # Independent published analyses of the genuine PersuasionForGood conversations.
    url='https://raw.githubusercontent.com/persuasion-nlp/persuasion-strategies/main/full_dialog_with_all_analysis.csv'
    resp=requests.get(url,timeout=24)
    resp.raise_for_status()
    df=pd.read_csv(io.BytesIO(resp.content),low_memory=False)
    if not {'Unit','B4','B2'}.issubset(df.columns):
        raise ValueError('Expected columns missing')
    df=df[df['B4'].astype(str).isin(['0','0.0'])].copy()
    df=df[df['Unit'].notna() & (df['Unit'].astype(str).str.len()>15)]
    return df

def kpi(name,value,helptext):
    st.metric(name,value,help=helptext)

def evidence_notice(pl):
    st.caption('Źródło: GoEmotions, Google Research, oficjalny test (anotacje ludzkie). Nie jest to reprezentatywna próba populacji.' if pl else 'Source: Google Research GoEmotions, official test split (human annotations). Not a population-representative sample.')

def render(section,df,real,pl,tr,emo,train_classifier):
    if section=='glossary':
        st.header(tr('Methods explained before experiments','Metody wyjaśnione przed eksperymentami'))
        definitions=[
            ('Codebook / klucz kodowy','A written set of category definitions, inclusion/exclusion rules, and examples.','Pisany zestaw definicji kategorii, zasad włączania/wykluczania oraz przykładów.'),
            ('Agreement / zgodność','Whether independent coders assign the same labels to the same text. Disagreement is informative, not a failure.','Czy niezależni koderzy przypisują te same etykiety temu samemu tekstowi. Rozbieżność to cenna informacja.'),
            ('TF-IDF','Turns words into numeric features: frequent in one text but rare across texts usually receive more weight.','Zamienia słowa na cechy liczbowe; wyrazy częste w danym tekście, lecz rzadkie w innych, otrzymują większą wagę.'),
            ('Unigram / 1-gram','A single token, e.g., good.','Jedno słowo, np. good.'),
            ('Bigram / 2-gram','Two consecutive tokens, e.g., not good.','Dwa kolejne słowa, np. not good.'),
            ('Logistic regression','An algorithm estimating a score for the presence of a selected label; not linear regression and not BERT.','Algorytm obliczający wynik dla wystąpienia etykiety; nie jest to regresja liniowa ani BERT.'),
            ('One-vs-rest','Train a separate yes/no classifier for each label: fear versus not fear, joy versus not joy, etc.','Dla każdej etykiety uczymy osobny model tak/nie: strach kontra nie-strach, radość kontra nie-radość itd.'),
            ('TP','Correct detection of an emotion.','Prawidłowe wykrycie danej emocji.'),
            ('FP','False alarm: predicts the emotion although the reference says it is absent.','Fałszywy alarm: model wykrył emocję, której nie ma w etykiecie referencyjnej.'),
            ('FN','Missed case: reference has the emotion, model does not detect it.','Przeoczenie: emocja jest w anotacji, model jej nie wykrył.'),
            ('TN','Correct rejection: the emotion is absent and the model says absent.','Poprawne niewykrycie: emocja nie występuje i model jej nie wskazuje.'),
            ('Precision','Among predictions of fear, what share are correct? TP/(TP+FP).','Spośród wszystkich przewidzianych przypadków strachu jaka część była poprawna? TP/(TP+FP).'),
            ('Recall','Among all reference fear cases, what share did we detect? TP/(TP+FN).','Spośród wszystkich rzeczywistych przypadków strachu ile model znalazł? TP/(TP+FN).'),
            ('F1','Harmonic mean of precision and recall; penalizes poor performance on either.','Średnia harmoniczna precision i recall; spada, gdy którakolwiek z miar jest niska.'),
            ('Jaccard overlap','Common labels divided by all distinct labels; not a full inter-rater reliability analysis.','Liczba wspólnych etykiet podzielona przez liczbę wszystkich różnych etykiet; to nie jest pełna miara rzetelności kodowania.')]
        for name,en,po in definitions:
            with st.expander(name):st.write(po if pl else en)
        st.markdown('**'+tr('Worked example','Przykład obliczeniowy')+'**')
        st.code('TP=32   FP=8   FN=12   TN=48\nPrecision=32/(32+8)=0.800\nRecall=32/(32+12)=0.727\nF1=2*0.800*0.727/(0.800+0.727)=0.762')
        st.info(tr('These quantities are about predictions versus a reference annotation, not about how emotional a person really felt.','To miary zgodności predykcji z anotacją referencyjną, a nie siły prawdziwych uczuć osoby.'))
    elif section=='real_lab':
        st.header(tr('Real-data exploration laboratory','Laboratorium eksploracji prawdziwych danych'))
        if not real:
            st.error(tr('Official GoEmotions data currently unavailable. We will not invent source results.','Brak połączenia z oficjalnymi danymi GoEmotions. Nie pokazujemy fikcyjnych wyników.'));return
        evidence_notice(pl)
        labels=Counter(x for row in df.labels for x in row)
        choices=[x for x,v in labels.most_common()]
        target=st.selectbox(tr('Choose an emotion to investigate','Wybierz emocję do badania'),choices,format_func=emo)
        mincount=st.slider(tr('Minimum number of annotations per comment','Minimalna liczba etykiet na komentarz'),1,4,1)
        mask=df.labels.apply(lambda x:target in x and len(x)>=mincount)
        sub=df[mask]
        a,b,c=st.columns(3)
        a.metric(tr('Comments with chosen label','Komentarze z daną etykietą'),len(sub))
        b.metric(tr('Share of test comments','Odsetek komentarzy testowych'),f'{100*len(sub)/len(df):.1f}%')
        c.metric(tr('Other labels also present','Komentarze z dodatkowymi etykietami'),int(sub.labels.map(lambda x:len(x)>1).sum()))
        cooccurrence=pd.DataFrame([{'label':emo(k),'count':v} for k,v in Counter(e for row in sub.labels for e in row if e!=target).most_common(10)])
        if len(cooccurrence):st.bar_chart(cooccurrence.set_index('label'))
        else:st.info(tr('No additional labels in the current filtered subset.','Brak dodatkowych etykiet po zastosowaniu filtra.'))
        if len(sub):
            i=st.slider(tr('Choose a real comment','Wybierz rzeczywisty komentarz'),0,min(len(sub)-1,250),0)
            ex=sub.iloc[i]
            st.info(ex['text'])
            with st.expander(tr('Human annotations: reveal after your interpretation','Anotacje ludzkie: odkryj po własnej interpretacji')):
                st.write(', '.join(emo(x) for x in ex.labels))
        st.markdown('**'+tr('Research task','Zadanie badawcze')+'**')
        st.write(tr('Increase the minimum labels from 1 to 2 or 3. Does the proportion of mixed labels change? Why does a filter change the denominator and interpretation?','Zwiększ minimalną liczbę etykiet z 1 do 2 lub 3. Jak zmienia się udział wypowiedzi wieloetykietowych? Dlaczego filtr zmienia podstawę porównania i interpretację?'))
        st.text_area(tr('Your interpretation (not saved)','Twoja interpretacja (bez zapisu)'),key='real_interpret')
    elif section=='agreement_lab':
        st.header(tr('Codebook and agreement studio','Studio klucza kodowego i zgodności'))
        st.write(tr('First write a rule, then classify text; finally compare coders. This prevents categories being changed merely to fit expected results.','Najpierw zapisz regułę, potem koduj tekst; na końcu porównaj koderów. Dzięki temu nie dopasowujesz definicji do oczekiwanych wyników.'))
        st.text_area(tr('Codebook: when does fear count?','Klucz kodowy: kiedy przypisać strach?'),value=tr('Include explicit fear or credible worry; exclude generic criticism unless the emotional interpretation is justified.','Włącz wyraźny strach lub uzasadnioną obawę; wyklucz samą krytykę bez podstaw do interpretacji emocjonalnej.'),height=85)
        if real:
            idx=st.slider(tr('Real GoEmotions comment index','Indeks rzeczywistego komentarza'),0,min(len(df)-1,400),42)
            row=df.iloc[idx];st.info(row.text)
        else:st.warning(tr('Real GoEmotions unavailable: this step needs a dataset.','Brak danych GoEmotions: krok wymaga zbioru.'))
        from content import EMOTIONS
        a=st.multiselect(tr('Coder A','Koder A'),EMOTIONS,default=['fear','nervousness'],format_func=emo)
        b=st.multiselect(tr('Coder B','Koder B'),EMOTIONS,default=['fear','surprise'],format_func=emo)
        intersection=set(a)&set(b); union=set(a)|set(b)
        aa,bb,cc=st.columns(3)
        aa.metric(tr('Shared labels','Wspólne etykiety'),len(intersection));bb.metric(tr('All distinct labels','Wszystkie różne etykiety'),len(union));cc.metric('Jaccard',f'{len(intersection)/len(union) if union else 1:.2f}')
        st.latex(r'J(A,B)=\frac{|A\cap B|}{|A\cup B|}')
        if real:
            with st.expander(tr('Reveal source annotation (not a perfect truth)','Pokaż etykiety źródłowe (nie nieomylna prawda)')): st.write(', '.join(emo(x) for x in row.labels))
        st.info(tr('For a research project: use independent coders, a pilot sample, clear exclusion rules, an adjudication protocol and a suitable reliability statistic.','W badaniu: niezależni koderzy, pilotaż, jasne zasady wykluczania, procedura rozstrzygania sporów i odpowiednia miara rzetelności.'))
    elif section=='threshold_lab':
        st.header(tr('Threshold experiment on genuine test labels','Eksperyment z progiem na rzeczywistych etykietach'))
        if not real:st.error(tr('Requires the official GoEmotions dataset.','Wymaga oficjalnego zbioru GoEmotions.'));return
        st.write(tr('Train a small teaching classifier on the official training split and evaluate on unseen test comments. The numeric values below are computed, not invented.','Wytrenuj prosty model na oficjalnej części treningowej i oceniaj go na niewidzianych komentarzach testowych. Poniższe wartości są obliczane, nie wymyślane.'))
        if st.button(tr('Train/load baseline model','Wytrenuj/wczytaj model bazowy'),key='train_experiment'):
            try:
                with st.spinner(tr('Calculating...','Obliczanie...')):train_classifier()
                st.session_state['baseline_experiment_ready']=True
            except Exception as e:st.error(str(e)[:180])
        if not st.session_state.get('baseline_experiment_ready',False):return
        vec,clf,labelset=train_classifier()
        target=st.selectbox(tr('Emotion','Emocja'),labelset,format_func=emo)
        n=st.slider(tr('Test comments to evaluate','Liczba komentarzy testowych'),100, min(1500,len(df)),500,100)
        threshold=st.slider(tr('Decision threshold','Próg decyzji'),0.05,0.95,0.50,0.05)
        subset=df.iloc[:n];probs=clf.predict_proba(vec.transform(subset.text))[:,labelset.index(target)]
        gold=np.asarray([target in x for x in subset.labels]);pred=probs>=threshold
        tp=int((pred & gold).sum());fp=int((pred & ~gold).sum());fn=int((~pred & gold).sum());tn=int((~pred & ~gold).sum())
        prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.;f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        c1,c2,c3=st.columns(3);c1.metric('Precision',f'{prec:.3f}');c2.metric('Recall',f'{rec:.3f}');c3.metric('F1',f'{f1:.3f}')
        st.dataframe(pd.DataFrame([[tp,fn],[fp,tn]],index=[tr('Actually labelled','Etykieta występuje'),tr('Actually absent','Brak etykiety')],columns=[tr('Predicted present','Przewidziano obecność'),tr('Predicted absent','Przewidziano brak')]),use_container_width=True)
        st.bar_chart(pd.DataFrame({'count':[tp,fp,fn,tn]},index=['TP','FP','FN','TN']))
        st.info(tr('Raise the threshold: there will usually be fewer positive predictions (often higher precision, lower recall). Verify rather than assume: small samples can behave differently.','Podnieś próg: zazwyczaj będzie mniej wskazań dodatnich (często wzrośnie precision, a spadnie recall). Sprawdź to zamiast zakładać — małe próby mogą zachowywać się inaczej.'))
        st.markdown('**'+tr('Real text intervention: what if we changed the message?','Interwencja na tekście: co jeśli zmienimy wiadomość?')+'**')
        original=df.iloc[4]['text']
        rewritten=st.text_area(tr('Edit this genuine GoEmotions comment; your edited version is a simulation','Zmień rzeczywisty komentarz GoEmotions; zmodyfikowana wersja jest symulacją'),value=original,key='model_real_edit')
        oldscore=float(clf.predict_proba(vec.transform([original]))[0,labelset.index(target)])
        newscore=float(clf.predict_proba(vec.transform([rewritten]))[0,labelset.index(target)])
        st.bar_chart(pd.DataFrame({'score':[oldscore,newscore]},index=[tr('Original real text','Oryginalny tekst'),tr('Your modified version','Twoja wersja')]))
        st.caption(tr('These are model scores, not changes in an actual person’s emotional state. Original content comes from the GoEmotions test set.','To wyniki modelu, nie zmiany rzeczywistych emocji człowieka. Oryginał pochodzi ze zbioru testowego GoEmotions.'))
        st.caption(tr('Baseline: TF-IDF (1-2 grams) + separate logistic classifier for each emotion. Trained on training rows; tested on unseen test rows. Scores are not necessarily calibrated.','Model bazowy: TF-IDF (1–2-gramy) + osobny klasyfikator logistyczny dla każdej emocji. Uczony na train; badany na test. Wyniki nie muszą być skalibrowane.'))
    elif section=='persuasion_lab':
        st.header(tr('Persuasion laboratory: six classic Cialdini principles','Laboratorium perswazji: sześć klasycznych zasad Cialdiniego'))
        st.write(tr('Learn principles first, then code messages from a real persuasion corpus. The corpus does NOT provide verified Cialdini labels: your labels are hypotheses to be tested, not a gold standard.','Najpierw poznaj zasady, potem koduj wypowiedzi z prawdziwego korpusu perswazyjnego. Korpus NIE zawiera zweryfikowanych etykiet Cialdiniego: Twoje oznaczenia są hipotezami, a nie wzorcem poprawnych odpowiedzi.'))
        for key,po,en,explain_pl,explain_en,example in SIX:
            with st.expander((po+' / '+en) if pl else en):
                st.write(explain_pl if pl else explain_en)
                st.caption(tr('Illustrative phrase (invented, not from dataset): ','Przykład autorski, nie ze zbioru: ')+example)
        st.caption(tr('Cialdini later added a seventh principle, Unity; this lab intentionally uses the classic six.','Cialdini dodał później siódmą zasadę — Unity (jedność); tutaj stosujemy klasyczne sześć.'))
        try:
            corpus=fetch_persuasion()
        except Exception as e:
            st.warning(tr('Cannot load public PersuasionForGood text right now. No invented text will be presented as real data.','Nie można teraz pobrać publicznych wypowiedzi PersuasionForGood. Nie pokażemy fikcyjnych tekstów jako prawdziwych danych.'))
            st.caption(str(e)[:130]);return
        st.caption(tr(f'Genuine published dialogue turns; current persuader-turn subset: {len(corpus)}. Original human conversation text; some later strategy annotations are AI-generated and NOT a Cialdini codebook.',f'Rzeczywiste tury opublikowanych rozmów: {len(corpus)} wypowiedzi perswadujących. Oryginalne rozmowy ludzkie; późniejsze etykiety strategii były m.in. generowane przez AI i NIE są kluczem Cialdiniego.'))
        target=st.selectbox(tr('Choose a real dialogue turn','Wybierz rzeczywistą wypowiedź'),range(min(len(corpus),300)),format_func=lambda i:f'{i+1} · {str(corpus.iloc[i]["Unit"])[:62]}')
        row=corpus.iloc[target]
        st.info(str(row['Unit']))
        selected=st.multiselect(tr('Which Cialdini principles do you see? Select only if evidence is explicit.','Które zasady Cialdiniego rozpoznajesz? Wskaż tylko przy wyraźnych dowodach.'),[x[0] for x in SIX],format_func=lambda k: next((p if pl else e for q,p,e,*_ in SIX if q==k),k))
        st.text_area(tr('Quote the words that support your decision and mention uncertainty.','Zacytuj fragment uzasadniający decyzję i wskaż niepewność.'),key='persuasion_why')
        st.markdown('**'+tr('Counterfactual experiment','Eksperyment kontrfaktyczny')+'**')
        st.write(tr('Rewrite this real message by adding (or removing) an authority cue. Compare your two versions. You changed an invented variant of real text, NOT the original dataset.','Przepisz tę rzeczywistą wypowiedź, dodając (lub usuwając) odwołanie do autorytetu. Porównaj wersje. Modyfikujesz własny wariant tekstu, a NIE oryginalny zbiór.'))
        st.text_area(tr('Your rewritten variant (not saved)','Twój zmodyfikowany wariant (bez zapisu)'),key='persuasion_rewrite')
        st.warning(tr('A message may have none, one or several principles. The corpus does not establish that any principle caused the donation.','Wypowiedź może nie zawierać żadnej zasady lub zawierać kilka. Korpus nie dowodzi, że któraś zasada spowodowała darowiznę.'))
    elif section=='final_real':
        st.header(tr('Final research challenge: use real evidence','Końcowe wyzwanie badawcze: pracuj na prawdziwych danych'))
        if not real:st.error(tr('Real GoEmotions dataset is required; cannot complete without it.','To zadanie wymaga prawdziwych danych GoEmotions.'));return
        st.write(tr('Research question: How do fear and gratitude differ in the actual GoEmotions test split?','Pytanie badawcze: Czym różnią się strach i wdzięczność w rzeczywistym zbiorze testowym GoEmotions?'))
        labels=['fear','gratitude','joy','anger','neutral','surprise']
        selections=st.multiselect(tr('Compare categories','Porównaj kategorie'),labels,default=['fear','gratitude'],format_func=emo)
        if len(selections)<2:st.info(tr('Select at least two emotions.','Wybierz co najmniej dwie emocje.'));return
        N=len(df);counts={e:sum(e in x for x in df.labels) for e in selections}
        chart=pd.DataFrame({'emotion':[emo(x) for x in selections],'count':[counts[x] for x in selections],'share_%':[round(100*counts[x]/N,2) for x in selections]})
        st.bar_chart(chart.set_index('emotion')[['count']]);st.dataframe(chart,hide_index=True,use_container_width=True)
        st.write(tr('**Simulation:** suppose we selected only texts with 2+ labels, or sampled 300 comments. How might prevalence and uncertainty change? Use the controls below to verify.','**Symulacja:** co się stanie, gdy wybierzemy tylko teksty z co najmniej dwiema etykietami albo próbę 300 komentarzy? Sprawdź zmianę częstości i niepewności za pomocą kontrolek.'))
        minlabels=st.slider(tr('At least this many source labels','Co najmniej tyle etykiet źródłowych'),1,4,1,key='final_nlabels')
        sample_size=st.slider(tr('Sample size','Liczebność próby'),100,min(1500,N),500,100,key='final_sample')
        subset=df[df.labels.map(len)>=minlabels]
        if len(subset)>=100:
            sample=subset.sample(n=min(sample_size,len(subset)),random_state=17)
            st.dataframe(pd.DataFrame({'category':[emo(x) for x in selections],tr('Share in filtered sample (%)','Udział w filtrowanej próbie (%)'):[round(100*sum(x in z for z in sample.labels)/len(sample),2) for x in selections]}),hide_index=True,use_container_width=True)
            st.caption(tr('Fixed random seed 17 ensures reproducibility; sampling variation and selection bias still matter.','Stałe ziarno losowe 17 zapewnia powtarzalność; zmienność próbkowania i stronniczość doboru nadal mają znaczenie.'))
        st.text_area(tr('State one empirical finding, one design limitation, and one next step.','Zapisz jeden wynik empiryczny, jedno ograniczenie projektu i jeden kolejny krok.'),height=125,key='final_real_answer')
        st.info(tr('The two percentages may sum beyond 100% because the annotations are multi-label. Correlation does not show why writers expressed emotions.','Suma procentów może przekroczyć 100%, ponieważ etykiety mogą współwystępować. Zależność nie wyjaśnia przyczyn emocjonalnych wypowiedzi.'))
