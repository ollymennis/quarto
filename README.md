# shakesphere

semantic search embeddings visualizer

shakespeare plays and sonnets are embedded using cohere's embed-v4.0 model and rendered as interactive particle clusters.

type anything into the search field — modern slang, a feeling, a question — and the engine finds the closest shakespeare passages by meaning.


https://github.com/user-attachments/assets/4b997116-cce8-4969-bf55-59dae285277b

## how it works

- each passage is a cluster of soft particles, color-coded by theme
- pca reduces 1536-dimensional embeddings down to 3 spatial dimensions
- cosine similarity measures how close your query is to each passage
- query expansion wraps short queries with context for better matching
- a wireframe globe marks where your query lands in the space

## setup

```
python3 embed.py
```

this embeds all passages from `passages.json` using your cohere api key (from `.env` or environment) and outputs `corpus.json`.

then open `index.html` in a browser or serve it:

```
python3 -m http.server 8080
```

## controls

- **type** — enter a query, press enter to search
- **hover** — reveals the source of each cluster
- **click** — select/deselect clusters to isolate them
- **drag** — orbit the 3d space
- **scroll** — zoom in and out
- **escape** — clear the current query

## stack

- three.js — 3d rendering, particles, post-processing
- cohere embed-v4.0 — semantic embeddings
- gsap — animation
- pca via power iteration — dimensionality reduction
- vanilla js — everything else

<img width="700" alt="Screenshot 2026-03-18 at 6 45 46 PM" src="https://github.com/user-attachments/assets/1207c2bf-d355-45a7-bf93-161ba956ad09" />
