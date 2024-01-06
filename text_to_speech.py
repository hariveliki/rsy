import random
from pathlib import Path
from openai import OpenAI
client = OpenAI()

definition_and_application = """
- Wie würdest du Recommender Systeme / Empfehlungsdienste in einem Satz beschreiben?→Recommender Systeme sind digitale Werkzeuge, die automatisiert personalisierte Empfehlungen oder Vorhersagen basierend auf den Präferenzen und dem Verhalten der Nutzer erstellen.
- Für welche analytischen Schlüssel-Aufgaben werden Recommender Systems eingesetzt?→Recommender Systems helfen bei der Entscheidungsfindung in Bereichen mit großer Auswahl, wie Online-Shopping, Film- und Musikdiensten, Buchempfehlungen und sozialen Medien, indem sie die Informationsflut filtern und relevante, personalisierte Inhalte oder Produkte vorschlagen.
- Wie hängen Vorhersage von Produkt-Ratings und Empfehlungen von Produkten bei Recommenders zusammen?→In Recommender Systemen werden vorhergesagte Produkt-Ratings genutzt, um Produkte zu empfehlen, die wahrscheinlich gut bewertet und vom Nutzer geschätzt werden, basierend auf historischen Daten und Nutzerinteraktionen.
- Was versteht man unter einem Hybrid Recommender System (mach ein sinnvolles Beispiel)?→Ein Hybrid Recommender System kombiniert verschiedene Empfehlungsansätze, wie z.B. kollaboratives und inhaltsbasiertes Filtern. Ein Beispiel ist ein Filmempfehlungssystem, das sowohl Nutzerbewertungen (kollaboratives Filtern) als auch inhaltliche Merkmale der Filme (inhaltsbasiertes Filtern) berücksichtigt.
- Welche Typen von Hybrid Recommender Systemen kann man unterscheiden?→Hybridansätze in Recommender Systemen umfassen gewichtete Hybridsysteme (Kombination verschiedener Vorhersagen durch Gewichtung), Switching Hybridsysteme (Auswahl basierend auf Situation oder Kontext), Feature-Kombination (Integration von Merkmalen verschiedener Ansätze), Meta-level Hybridsysteme (ein Ansatz generiert Merkmale für einen anderen) und kaskadierende Hybridsysteme (ein Ansatz filtert Kandidaten, ein anderer rangiert sie).
"""

input_data = """
- Welche Arten von Daten brauchen alle Recommender Systems?
    - Benutzerdaten (wie demografische Informationen und Nutzerprofile)
    - Interaktionsdaten (wie Kaufhistorie, Browsing-Verhalten, Ratings)
    - Produktinformationen (wie Beschreibungen, Kategorien, Preise).
- Was ist der Unterschied von impliziten und expliziten Kundenratings?
    - Explizite sind direkte Bewertungen eines Produkts oder einer Dienstleistung (z.B. Sternebewertungen, Likes).
    - Implizite werden abgeleitet, wie z.B. Kaufhistorie, Verweildauer auf einer Seite, Klickverhalten.
- Welche Arten von expliziten Kundenratings gibt es, was ist bei deren Interpretation zu beachten?
    - Sternebewertungen
    - Likes/Dislikes
    - schriftliche Rezensionen
    - Ranglisten
    - Subjektiv, beeinflusst durch persönlichen Präferenzen, Stimmungen, Erwartungen.
- Welche Möglichkeiten gibt es implizite Kundenratings für Recommender Systeme zu gewinnen?
    - Klickrate
    - Kaufhistorie
    - Verweildauer auf einer Webseite
    - Suchhistorie
    - Häufigkeit, mit der ein Artikel angesehen wird.
- Welche Schwierigkeiten gibt es bei der Verwendung und Interpretation von expliziten bzw. impliziten Ratings in Recommender Systemen?
    - Explizit ⇒ Verzerrungen durch extreme Bewertungen, unterschiedliche Standards für Bewertungen. 
    - Implizit ⇒ Mehrdeutig, Nutzerverhalten spiegelt nicht immer ein klares Interesse wieder, schwierig präzise Präferenzen abzuleiten."""

simple_recommender_systems = """
- Welches ist die plausibelste / beste nicht-personalisierte Produktempfehlung?→Basierend auf aggregierten Daten, meistverkaufte Produkte, am höchsten bewertete Produkte.
- Wie werden für den Personalized Mean Recommender Ratingvorhersagen erzeugt?→Durch Anpassung des durchschnittlichen Ratings eines Nutzers mit den Ratings, die andere Nutzer für dasselbe Produkt abgegeben haben.
- Welche Bedeutung hat die Dämpfungskonstante beta für das Personalized Mean Modell?→Die Dämpfungskonstante 𝛽𝑢 im Personalized Mean Modell dient dazu, den Einfluss der Anzahl der vom Nutzer bewerteten Items auf das endgültige Rating zu regulieren. Eine höhere Konstante 𝛽𝑢 verringert den Einfluss der Anzahl der Bewertungen und sorgt so für eine stärkere Glättung der Vorhersagen.
- Inwiefern sind Empfehlungen, welche mit Association Rules hergeleitet werden, personalisiert?→Sie sind nicht streng personalisiert, da sie nicht auf den individuellen Präferenzen eines einzelnen Nutzers basieren, sondern auf der Analyse von Transaktionsdaten und Mustern, die aus der gesamten Nutzerbasis abgeleitet werden.
- In welchen Situationen eignet sich Confidence als Metrik für Association Rule Empfehlungen nicht?→Confidence als Metrik für Association Rule Empfehlungen ist nicht geeignet, wenn die Popularität der Produkte stark variiert. Bei sehr beliebten Produkten kann die Confidence irreführend hoch sein, selbst wenn keine wirkliche Assoziation zwischen den Produkten besteht, da das beliebte Produkt ohnehin von vielen Nutzern gekauft wird.
- Wie ist der Lift einer Association Rule "Wenn Kunde Produkt 𝑖 kauft, dann kauft er auch Produkt 𝑗" zu interpretieren?→Der Lift einer Association Rule gibt an, wie viel wahrscheinlicher es ist, dass Produkt 𝑗 gekauft wird, wenn Produkt 𝑖 bereits gekauft wurde, im Vergleich zur Wahrscheinlichkeit, dass Produkt 𝑗 unabhängig von Produkt 𝑖 gekauft wird. Ein Lift größer als 1 deutet auf eine positive Assoziation hin, während ein Lift kleiner als 1 eine negative Assoziation anzeigt.
"""

content_based_recommender = """
- Welche Arten von Informationen verwenden Content-based Recommender Systeme?
    - Textbeschreibungen
    - Schlagworte
    - Genres
    - Autoren bei Büchern
    - Schauspieler und Regisseure bei Filmen
    - spezifische Produkteigenschaften wie Farbe, Größe, und Stil.
- Welche Komponenten verwenden Content-based Recommender Systeme?
    - (1) ein Feature-Extraktionsmodul, das relevante Merkmale aus den Produktinhalten extrahiert; 
    - (2) ein Profil-Lernmodul, das Nutzerpräferenzen basierend auf Interaktionen mit Produkten lernt;
    - (3) ein Empfehlungsmodul, das Produkte vorschlägt, die zu den gelernten Nutzerpräferenzen passen.
- Welches sind Vorteile bzw. Nachteile von Content-based Recommender Systemen?
    - Vorteile: Sie sind unabhängig von anderen Nutzern und können daher auch für neue Produkte oder Nutzer ohne vorherige Bewertungen Empfehlungen abgeben.
    - Nachteile: Sie sind auf die verfügbaren Inhaltsinformationen beschränkt und neigen dazu, Nutzern ähnliche Produkte zu empfehlen, wodurch die Vielfalt der Empfehlungen eingeschränkt wird.
- Was versteht man im Bereich Text-Repräsentation unter dem Bag-of-Words und Vektorraum Modell?
    - Bag-of-Words-Modell ist ein Ansatz zur Text-Repräsentation, bei dem ein Text als eine Sammlung von Wörtern ohne Berücksichtigung von Grammatik oder Wortreihenfolge dargestellt wird.
    - Das Vektorraum-Modell wandelt diese Sammlung in einen Vektor um, wobei jedes Wort im Vektorraum eine Dimension repräsentiert.
- Welche Schritte werden im Bag-of-Words Modell typischerweise implementiert?
    - (1) Tokenisierung des Textes in einzelne Wörter;
    - (2) Entfernung von Stop-Wörtern;
    - (3) Anwendung von Stemming oder Lemmatisierung zur Reduzierung von Wörtern auf ihre Grundform;
    - (4) Umwandlung des Textes in einen Vektor, wobei die Häufigkeit jedes Wortes berücksichtigt wird.
- Was ist die Bedeutung von TF-IDF für Content-based Recommender Systems?→TF-IDF (Term Frequency-Inverse Document Frequency) ist eine Methode, um die Relevanz eines Wortes in einem Dokumenten zu bewerten. In Content-based Recommender Systems wird TF-IDF verwendet, um die wichtigsten Wörter in den Produktbeschreibungen zu identifizieren und somit eine bessere Unterscheidung zwischen relevanten und irrelevanten Produkten zu ermöglichen.
- Wie ist die TF-IDF Formel zu interpretieren?↓
    - TF (Term Frequency) ist die Häufigkeit eines Wortes in einem Dokument
    - IDF (Inverse Document Frequency) ist die Anzahl aller Dokumente über die Anzahl der Dokumente in denen das Wort vorkommt.
    - TF-IDF ist hoch falls das Wort in einem Dokument of vorkommt aber nicht in vielen Dokumenten.
    - tfidf(t, d, D) = tf(t, d) * idf(t, D)
"""

collaborative_filtering_recommender = """
- Worin unterscheiden sich Content-based von Collaborative Filtering Recommender Systemen?
    - Content-based basiert auf den Eigenschaften der Produkte und Empfehlungen basieren auf der Ähnlichkeit zwischen Produkten.
    - Collaborative Filtering hingegen basiert auf den Bewertungen und Präferenzen von Nutzern; Empfehlungen werden aufgrund der Ähnlichkeiten zwischen Nutzern oder zwischen Produkten, basierend auf Nutzerbewertungen, generiert.
- Welches sind Vorteile bzw. Nachteile von Collaborative Filtering Recommender Systemen?
    - Vorteile: keine detaillierten Informationen über die Produkte werden benötigt und überraschende Empfehlungen für Nutzer abseits der bekannten Präferenzen. 
    - Nachteile: Cold-Start-Problem (Schwierigkeiten bei neuen Nutzern oder Produkten ohne Bewertungen) und Probleme bei der Skalierbarkeit mit wachsender Nutzer- und Produktanzahl.
- Welches sind die Unterschiede von Model-based vs Memory-based Collaborative Filtering Recommender Systemen?
    - Memory-based Ansätze (auch als heuristische Ansätze bekannt) verwenden die gesamten Daten zur Generierung von Empfehlungen und basieren auf Ähnlichkeitsberechnungen zwischen Nutzern oder Produkten.
    - Model-based entwickelt ein Modell basierend auf den Trainingsdaten, das dann zur Vorhersage von Bewertungen verwendet wird. Model-based Ansätze sind in der Regel effizienter und skalierbarer.
- Wozu dient bei UBCF und IBCF der Parameter 𝑘 (für die Nachbarschaftsgrösse) und bei SVD der Parameter 𝑘 (für den Rang)?
    - Bei User-Based Collaborative Filtering (UBCF) und Item-Based Collaborative Filtering (IBCF) gibt der Parameter 𝑘 die Anzahl der nächstgelegenen Nachbarn (User oder Items) an, die für die Vorhersage berücksichtigt werden.
    - Bei der Singular Value Decomposition (SVD) bezieht sich der Parameter 𝑘 auf die Anzahl der latenten Faktoren (den Rang der Matrix), die in dem Modell verwendet werden, um die Nutzer- und Item-Interaktionen zu repräsentieren.
"""

memory_collaborative_recommender = """
- Wie beschreibst du die Formeln für die Rating-Vorhersage mit UBCF bzw. IBCF möglichst präzise in Worten?
    - UBCF: Die vorhergesagte Bewertung p_{u,i} für einen Nutzer u und ein Item i ist das durchschnittliche Rating des Nutzers r_u plus die gewichtete Summe der Abweichungen der Ratings der ähnlichsten Nutzer (Nachbarn) für das Item i von deren Durchschnittsratings, geteilt durch die Summe der absoluten Ähnlichkeitswerte.
    - IBCF: Die vorhergesagte Bewertung p_{u,i} ist der Basiswert b_{u,i} (oft ein Durchschnittswert) plus die gewichtete Summe der Abweichungen der Ratings des Nutzers für ähnliche Items von diesem Basiswert, geteilt durch die Summe der absoluten Ähnlichkeitswerte der Items.
- Welche Methoden kennst du, um für UBCF die Anzahl Nachbarn zu steuern?
    - Feste Anzahl k der nächstgelegenen Nachbarn festlegen.
    - Einen Schwellenwert für die Ähnlichkeitswerte festlegen, um nur die relevantesten Nachbarn zu berücksichtigen.
    - Adaptive Methoden verwenden, die die Anzahl der Nachbarn basierend auf der spezifischen Situation oder den Eigenschaften der Daten anpassen.
- Welches sind die Vorteile bzw. Nachteile der Methoden zur Beschränkung der Anzahl Nachbarn bei UBCF, wie/wann werden sie eingesetzt und wo ist Vorsicht geboten?
    - Vorteile: Reduzierte Rechenlast und eine potenziell höhere Qualität der Empfehlungen durch Fokussierung auf die relevantesten Nachbarn. 
    - Nachteile: Verringerte Diversität der Empfehlungen und das Risiko von überangepassten Empfehlungen.
    - Wie/Wann: Um die Effizienz zu steigern und die Empfehlungsqualität zu verbessern. 
    - Vorsicht: Relevante Informationen können ausgeschlossen werden, was zu einer Beeinträchtigung der Empfehlungsgenauigkeit führen kann.
- Was versteht man unter variance weighting factor, unter significance weighting factor und unter case amplification und wie/wo werden diese eingesetzt?↓
    - Variance weighting factor wird verwendet, um die Gewichtung von Nachbarn basierend auf der Varianz ihrer Ratings zu justieren, sodass konstantere Bewerter stärker gewichtet werden. 
    - Significance weighting factor dient dazu, die Ähnlichkeit zwischen Nutzern oder Items zu modifizieren, basierend auf der Anzahl gemeinsamer Ratings, um zufällige Übereinstimmungen zu minimieren.
    - Case amplification verstärkt die Gewichtung von sehr ähnlichen Nutzern oder Items, indem es die Ähnlichkeitswerte potenziert.
"""

memory_collaborative_recommender_2 = """
- Richtig oder Falsch: “Wenn die minimale Ähnlichkeit für UBCF 𝑘 zu hoch gewählt wird, reduziert sich unter Umständen die Anzahl Produkte für die eine Vorhersage gemacht werden kann”→Richtig. Wenn diese Schwelle zu hoch gesetzt wird, bedeutet das, dass nur Nutzer mit sehr ähnlichen Interessen oder Bewertungen als relevant für die Vorhersagen angesehen werden. Dies führt dazu, dass für einige Produkte keine Nutzer gefunden werden, und somit keine Vorhersage gemacht werden kann.
- Richtig oder Falsch: “Wenn die Zahl der berücksichtigten Nachbarn 𝑘 bei UBCF zu gross gewählt wird, verschlechtern sich die Vorhersagen der Empfehlungen”
    - Teilweise richtig.
    - Eine zu große Anzahl an Nachbarn kann die Qualität der Vorhersagen beeinträchtigen, da weniger ähnliche Nutzer in die Berechnung einbezogen werden.
    - Andererseits kann eine zu geringe Anzahl an Nachbarn dazu führen, dass nicht genügend Informationen für genaue Vorhersagen zur Verfügung stehen.
- Richtig oder Falsch: “Für UBCF wird typischerweise die User-User Ähnlichkeitsmatrix über Nacht vorprozessiert (offline precomputation), um eine effizientere Berechnung von Empfehlungen im Tagesbetrieb zu gewährleisten”→Richtig. Dies ermöglicht eine schnellere Berechnung von Empfehlungen während des laufenden Betriebs. Dies ist besonders wichtig für Systeme mit vielen Nutzern und Produkten, um eine hohe Antwortgeschwindigkeit zu gewährleisten.
"""

similarity_measure = """
- Welche Ähnlichkeitsmasse finden bei Recommender Systems typischerweise Anwendung (beschreibe sie in Formeln)?
    - Cosine Similarity: Die Kosinusähnlichkeit misst den Kosinus des Winkels zwischen zwei Vektoren im Vektorraum. In Recommender Systemen sind diese Vektoren oft die Bewertungen von Nutzern oder die Eigenschaften von Items. Ein Wert von 1 bedeutet vollständige Übereinstimmung (gleiche Richtung), während 0 keine Übereinstimmung anzeigt.
    - Pearson Correlation Coefficient: Der Pearson-Korrelationskoeffizient misst den linearen Zusammenhang und gibt die Stärke und Richtung zwischen den Bewertungen zweier Nutzer oder Items an. Ein Wert von 1 bedeutet eine perfekte positive Korrelation, -1 eine perfekte negative Korrelation und 0 keine Korrelation.
    - Jaccard Similarity: Die Jaccard-Ähnlichkeit ist ein Maß für die Übereinstimmung zwischen zwei Datensätzen. Sie berechnet sich als die Schnittmenge geteilt durch die Größe ihrer Vereinigung. In binären Daten zeigt sie an, wie ähnlich zwei Nutzer oder Items hinsichtlich der Eigenschaften oder Interaktionen sind.
- Was versteht man unter Adjusted Cosine Similarity und worin besteht der Hauptunterschied gegenüber (a) der Cosine Similarity und (b) der Pearson Similarity (beschreibe diese in Formeln)?
    - Adjusted Cosine Similarity berücksichtigt die durchschnittliche Bewertung der Nutzer, um die individuellen Bewertungsstandards auszugleichen. Die Formel ist ähnlich der der Cosine Similarity, aber anstatt der Rohbewertungen werden die Abweichungen der Bewertungen vom jeweiligen Nutzerdurchschnitt verwendet.
    - Der Hauptunterschied zur (a) Cosine Similarity ist die Berücksichtigung der Nutzerdurchschnitte.
    - Der Hauptunterschied zu (b) Pearson Similarity ist die Item-Perspektive anstatt die Nutzer-Perspektive.
- Worauf ist bei der Berechnung von Ähnlichkeitsmassen für Sparse Matrices speziell zu achten?→Dass nur die Einträge berücksichtigt werden, für die beide Nutzer (oder Items) Bewertungen abgegeben haben. Dies verhindert Verzerrungen durch viele Nullwerte und hilft für eine effizientere  Berechnung.
"""

model_based_collaborative = """
- Was ist die grundlegende, intuitive Idee hinter der Verwendung von SVD basierten Recommender Systemen?→Ist die Reduktion der Komplexität und Dimensionalität der Bewertungsdaten. SVD identifiziert die zugrundeliegenden Muster in den Bewertungsdaten, indem es die großen und spärlichen Nutzer-Item-Bewertungsmatrizen in kleinere, dichtere Matrizen zerlegt, die die latenten Faktoren repräsentieren. Diese Faktoren spiegeln die verborgenen Präferenzen der Nutzer und Eigenschaften der Produkte wider.
- Worin liegt der Hauptgewinn des SVD-Algorithmus für Recommender Systeme?→Der Hauptgewinn des SVD-Algorithmus liegt in seiner Fähigkeit, die Schlüsselelemente der Nutzer-Item-Interaktionen zu erfassen, was zu genaueren Empfehlungen führt. Durch die Reduzierung der Dimensionalität werden die wesentlichen Muster hervorgehoben, wodurch das System effizienter arbeiten kann und gleichzeitig die Relevanz der Empfehlungen verbessert wird.
- Wo liegt die Schwierigkeit bei Verwendung des SVD-Algorithmus für Recommender Systeme?→In der Behandlung von fehlenden Werten. In der Praxis sind Bewertungsmatrizen oft spärlich besetzt, da nicht jeder Nutzer jedes Produkt bewertet. Die Standard-SVD kann jedoch nicht direkt auf Matrizen mit vielen fehlenden Werten angewendet werden.
- Wie kann die oben genannte Schwierigkeit der Verwendung von SVD für Recommender Systeme gelöst werden?→Durch Methoden wie Matrix Faktorisierung, bei der Modelle wie Funk-SVD oder Alternating Least Squares (ALS) verwendet werden. Diese Ansätze optimieren die Faktormatrizen, um die vorhandenen Bewertungen so genau wie möglich vorherzusagen, anstatt direkt mit einer vollständigen Matrix zu arbeiten.
- Welche Dimensionen hat die Repräsentation der Kunden- bzw. Produktpräferenz, wenn die Rating-Matrix für m Kunden und n Produkte mit SVD auf k Dimensionen reduziert wird?→Wenn eine Rating-Matrix für m Kunden und n Produkte mittels SVD auf k Dimensionen reduziert wird, resultiert dies in zwei Matrizen: Eine Nutzermatrix der Größe m x k und eine Produktmatrix der Größe n x k. Jeder Nutzer und jedes Produkt wird dabei durch einen Vektor mit k Dimensionen repräsentiert, der die latenten Präferenzen bzw. Eigenschaften widerspiegelt.
- Was versteht man bei der Berechnung von Rating Vorhersagen mit SVD unter Folding in?→Unter "Folding in" versteht man einen Prozess, bei dem neue Nutzer- oder Produktinformationen in das bestehende SVD-Modell integriert werden, ohne das gesamte Modell neu zu berechnen. Dabei werden die Bewertungen des neuen Nutzers oder Produkts verwendet, um dessen latente Faktoren zu bestimmen, indem sie mit den vorhandenen latenten Faktoren der SVD-Matrix kombiniert werden. Dies ermöglicht es, Vorhersagen für neue Nutzer oder Produkte effizient zu generieren.
"""

model_based_collaborative_2 = """
- Richtig oder Falsch: “Die Präferenz eines Kunden für ein bestimmtes Produkt ergibt sich bei SVD als gewichtete Summe der Kundenpräferenz für jedes Thema multipliziert mit der Relevanz des Produkts für das Thema”→Richtig. In SVD-basierten Recommender Systemen wird die Präferenz eines Kunden für ein bestimmtes Produkt in der Tat als gewichtete Summe der Kundenpräferenz für verschiedene latente Faktoren (häufig als 'Themen' bezeichnet) berechnet. Diese Präferenzen werden dann mit der Relevanz (oder dem Gewicht) jedes Produkts bezüglich dieser latenten Faktoren multipliziert. Dieser Ansatz ermöglicht es, die Beziehungen zwischen Kunden und Produkten in einem reduzierten, aber informativen latenten Raum zu erfassen.
- Richtig oder Falsch: “Die m x k dimensionale SVD-Repräsentation der Kundenpräferenz kann für die Berechnung von Ähnlichkeiten bzw. Konstruktion von Nachbarschaften verwendet werden.”→Richtig. Die durch SVD erzeugte m x k dimensionale Matrix, die die Kundenpräferenzen darstellt, kann genutzt werden, um Ähnlichkeiten zwischen den Nutzern zu berechnen. Diese Ähnlichkeitsberechnungen können dann dazu verwendet werden, Nachbarschaften zu konstruieren, indem ähnliche Nutzer zusammengefasst werden. Dies ermöglicht eine effektive Identifikation von Nutzergruppen mit ähnlichen Vorlieben oder Verhaltensweisen, was für personalisierte Empfehlungen in Recommender Systemen von großem Nutzen ist.
"""

voices = ["alloy", "echo", "fable", "nova", "shimmer"]
voice = random.choice(voices)
speech_file_path = Path(__file__).parent / f"out/model_based_collaborative_2_{voice}.mp3"
response = client.audio.speech.create(
    model = "tts-1-hd",
    voice = voice,
    input = model_based_collaborative_2
)
response.stream_to_file(speech_file_path)
