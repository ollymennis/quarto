#!/usr/bin/env python3
"""
Embed a passages.json file using Cohere's embed-v4.0 model.
Outputs corpus.json with embeddings included.

Usage:
  python3 embed.py                           # uses .env for API key
  COHERE_API_KEY=xxx python3 embed.py        # explicit key
  python3 embed.py --input my_passages.json  # custom input
"""

import json
import os
import sys
import urllib.request
import urllib.error
import time

def get_api_key():
    key = os.environ.get('COHERE_API_KEY', '')
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith('COHERE_API_KEY='):
                        key = line.strip().split('=', 1)[1].strip()
    if not key:
        print("Error: No COHERE_API_KEY found in environment or .env file")
        sys.exit(1)
    return key

def embed_batch(texts, api_key, input_type="search_document"):
    """Embed a batch of texts using Cohere embed-v4.0."""
    payload = json.dumps({
        "model": "embed-v4.0",
        "texts": texts,
        "input_type": input_type,
        "embedding_types": ["float"],
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.cohere.com/v2/embed',
        data=payload,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data['embeddings']['float']
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        print(f"API error {e.code}: {body}")
        sys.exit(1)

def main():
    input_file = 'passages.json'
    output_file = 'corpus.json'

    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--input' and i + 2 < len(sys.argv):
            input_file = sys.argv[i + 2]
        if arg == '--output' and i + 2 < len(sys.argv):
            output_file = sys.argv[i + 2]

    api_key = get_api_key()

    with open(os.path.join(os.path.dirname(__file__) or '.', input_file)) as f:
        data = json.load(f)

    passages = data['passages']
    texts = [p['text'] for p in passages]

    print(f"Embedding {len(texts)} passages using Cohere embed-v4.0...")

    # Batch in groups of 96 (API limit)
    batch_size = 96
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} texts...")
        embeddings = embed_batch(batch, api_key)
        all_embeddings.extend(embeddings)
        if i + batch_size < len(texts):
            time.sleep(0.5)  # Rate limit courtesy

    # Attach embeddings to passages
    for p, emb in zip(passages, all_embeddings):
        p['embedding'] = emb

    # Output
    output = {
        "meta": data['meta'],
        "categories": data['categories'],
        "passages": passages,
    }

    out_path = os.path.join(os.path.dirname(__file__) or '.', output_file)
    with open(out_path, 'w') as f:
        json.dump(output, f)

    dim = len(all_embeddings[0]) if all_embeddings else 0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done! {len(passages)} passages × {dim}D embeddings → {output_file} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
