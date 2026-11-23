
import sys

from src.search import search

query = " ".join(sys.argv[1:])

if not query:
    print("Please provide a search question.")
else:
    for r in search(query, top_k=3):
        print(f"\n[{r['score']}] {r['source']} (page {r['page']})")
        print(r["text"][:300], "...")
