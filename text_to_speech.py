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
- Welche Arten von Daten brauchen alle Recommender Systems?→Recommender Systems benötigen verschiedene Arten von Daten, darunter Benutzerdaten (wie demografische Informationen und Nutzerprofile), Interaktionsdaten (wie Kaufhistorie, Browsing-Verhalten, Ratings) und Produktinformationen (wie Beschreibungen, Kategorien, Preise).
- Was ist der Unterschied von impliziten und expliziten Kundenratings?→Explizite Ratings sind direkte Bewertungen eines Produkts oder einer Dienstleistung durch den Nutzer (z.B. Sternebewertungen, Likes). Implizite Ratings hingegen werden aus dem Verhalten der Nutzer abgeleitet, wie z.B. Kaufhistorie, Verweildauer auf einer Seite oder Klickverhalten.
- Welche Arten von expliziten Kundenratings gibt es, was ist bei deren Interpretation zu beachten?→Explizite Kundenratings können Sternebewertungen, Likes/Dislikes, schriftliche Rezensionen oder Ranglisten sein. Bei deren Interpretation ist zu beachten, dass sie subjektiv sind und von persönlichen Präferenzen, Stimmungen und Erwartungen des Nutzers beeinflusst werden können.
- Welche Möglichkeiten gibt es implizite Kundenratings für Recommender Systeme zu gewinnen?→Implizite Kundenratings können durch Analyse von Verhaltensdaten gewonnen werden, wie z.B. Klickrate, Kaufhistorie, Verweildauer auf einer Webseite, Suchhistorie oder die Häufigkeit, mit der ein Artikel angesehen wird.
- Welche Schwierigkeiten gibt es bei der Verwendung und Interpretation von expliziten bzw. impliziten Ratings in Recommender Systemen?→Bei expliziten Ratings kann es zu Verzerrungen kommen, da Nutzer möglicherweise nur extreme Bewertungen abgeben oder unterschiedliche Standards für Bewertungen anlegen. Implizite Ratings hingegen können mehrdeutig sein, da das Nutzerverhalten nicht immer ein klares Interesse widerspiegelt, und es kann schwierig sein, daraus präzise Präferenzen abzuleiten.
"""

speech_file_path = Path(__file__).parent / "out/input_data_hd_alloy.mp3"
response = client.audio.speech.create(
    model = "tts-1-hd",
    voice = "alloy",
    input = input_data
)
response.stream_to_file(speech_file_path)
