"""The "answer key" for evaluating the Leipzig RAG system.

Each entry is a question plus the factually correct ``ground_truth`` we expect.
The last item is intentionally out-of-domain: it checks that the system admits
it does not know instead of hallucinating.

A local 12B model is slow (~2 min per answer), so by default only three
representative questions are active: a historical one, a cultural/person one
and the out-of-domain check. Uncomment the rest for a full-scale evaluation.
"""

EVALUATION_DATA: list[dict[str, str]] = [
    {
        "question": "Wann fand die Völkerschlacht bei Leipzig statt und wer "
                    "kämpfte gegen wen?",
        "ground_truth": "Die Völkerschlacht bei Leipzig fand vom 16. bis 19. "
                        "Oktober 1813 statt. Die verbündeten Armeen von "
                        "Russland, Preußen, Österreich und Schweden besiegten "
                        "die Armee Napoleons. Sie gilt als eine der größten "
                        "Schlachten der Befreiungskriege.",
    },
    {
        "question": "Welche Verbindung hat Johann Sebastian Bach zu Leipzig?",
        "ground_truth": "Johann Sebastian Bach war ab 1723 bis zu seinem Tod "
                        "1750 Thomaskantor in Leipzig. Er leitete den "
                        "Thomanerchor an der Thomaskirche und schuf dort viele "
                        "seiner bedeutendsten Werke. Er ist in der Thomaskirche "
                        "begraben.",
    },
    {
        "question": "Wie hoch ist der Eiffelturm in Paris?",
        "ground_truth": "Die Wissensbasis über Leipzig enthält keine "
                        "Informationen über den Eiffelturm. Das System sollte "
                        "angeben, dass es diese Frage anhand der vorliegenden "
                        "Dokumente nicht beantworten kann.",
    },
    # --- Uncomment for a full evaluation (slower) ---
    # {
    #     "question": "Welche Rolle spielte die Nikolaikirche bei der "
    #                 "Friedlichen Revolution 1989?",
    #     "ground_truth": "Die Nikolaikirche in Leipzig war ein zentraler "
    #                     "Ausgangspunkt der Friedlichen Revolution. Dort fanden "
    #                     "die Friedensgebete statt, aus denen sich die "
    #                     "Montagsdemonstrationen entwickelten, die maßgeblich "
    #                     "zum Ende der DDR beitrugen.",
    # },
    # {
    #     "question": "Wann wurde die Universität Leipzig gegründet?",
    #     "ground_truth": "Die Universität Leipzig wurde 1409 gegründet und "
    #                     "zählt damit zu den ältesten Universitäten "
    #                     "Deutschlands.",
    # },
    # {
    #     "question": "Was ist das Völkerschlachtdenkmal und wann wurde es "
    #                 "eingeweiht?",
    #     "ground_truth": "Das Völkerschlachtdenkmal in Leipzig erinnert an die "
    #                     "Völkerschlacht von 1813. Es wurde 1913 zum 100. "
    #                     "Jahrestag eingeweiht und ist mit rund 91 Metern eines "
    #                     "der höchsten Denkmäler Europas.",
    # },
    # {
    #     "question": "Warum ist Auerbachs Keller so bekannt?",
    #     "ground_truth": "Auerbachs Keller ist eine der bekanntesten "
    #                     "Gaststätten Leipzigs in der Mädlerpassage. Berühmt "
    #                     "wurde sie vor allem durch Johann Wolfgang von Goethes "
    #                     "'Faust', in dem eine Szene in Auerbachs Keller spielt.",
    # },
]
