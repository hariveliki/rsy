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

speech_file_path = Path(__file__).parent / "out/collaborative_filtering_recommender_nova.mp3"
response = client.audio.speech.create(
    model = "tts-1-hd",
    voice = "nova",
    input = collaborative_filtering_recommender
)
response.stream_to_file(speech_file_path)
