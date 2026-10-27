---
title: BookMind Pro
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid book recommender — semantic search + collaborative filtering
---

# BookMind Pro — Hybrid Book Recommender

Search books by **meaning** ("a slow-burn mystery set in winter"), browse a genre, and get
**"books like this one"** recommendations.

- **Semantic search:** sentence embeddings (`all-MiniLM-L6-v2`) + FAISS over ~6.3k books
- **Collaborative filtering:** item-item k-NN trained on ~1.15M Book-Crossing ratings
- **Stack:** FastAPI · Pydantic · Sentence-Transformers · FAISS · scikit-learn · 480+ tests

Try it:
- **Search:** click the empty search box for popular books, or type a description
- **Recommend:** enter a title such as *Harry Potter and the Sorcerer's Stone*
- **API docs:** add `/docs` to the address
- **Health:** add `/health/ready` to the address

Free and local: no paid API keys. The source code and setup guide are in the project repository.
