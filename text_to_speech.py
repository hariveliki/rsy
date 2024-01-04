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

speech_file_path = Path(__file__).parent / "out/definition_and_application.mp3"
response = client.audio.speech.create(
    model = "tts-1",
    voice = "alloy",
    input = definition_and_application
)
response.stream_to_file(speech_file_path)
