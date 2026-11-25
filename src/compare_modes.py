from src.search import search

QUERIES = [
    "What causes global warming?",
    "history of Bengal",
    "বাংলার ইতিহাস",
    "Sundarbans mangrove forest",
]

for n, q in enumerate(QUERIES, start=1):
    print(f"\nQ{n}: {q if q.isascii() else '(Bangla question)'}")

    for mode in ("semantic", "keyword", "hybrid"):
        hits = [
            (r["source"], r["page"])
            for r in search(q, 3, mode=mode)
        ]
        print(f"  {mode:9}", hits)