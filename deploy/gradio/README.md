---
title: BookMind Pro
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {sdk_version}
python_version: "{python_version}"
app_file: app.py
pinned: false
short_description: Hybrid book recommender - semantic search and k-NN
---

# BookMind Pro — Hybrid Book Recommender

Search books by **meaning** ("a slow-burn mystery set in winter"), browse a genre, and get
**"books like this one"** recommendations.

- **Semantic search:** sentence embeddings (`all-MiniLM-L6-v2`) + FAISS over ~6.3k books
- **Collaborative filtering:** item-item k-NN trained on ~1.15M Book-Crossing ratings
- **Hybrid:** "Recommend" uses collaborative filtering when the title is in the ratings data,
  otherwise semantic similarity

This Space runs the project's FastAPI website (search, genre browsing, recommendations,
REST API and `/docs`); a Gradio version of the screens is at `/demo`. The full project also
includes an offline data pipeline, 500+ automated tests, Docker and CI/CD.
Everything is free and open source; no paid API keys.
