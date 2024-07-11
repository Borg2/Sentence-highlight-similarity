import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def read_text_files(directory):
    sentences = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                sentences.extend([sent.strip() for sent in text.split('. ') if sent.strip()])
    return sentences

# Directory containing the text files
directory = "text_files"

# Read sentences from text files 
sentences = read_text_files(directory)

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)

# Calculate cosine similarities
similarities = cosine_similarity(sentence_vectors)

average_similarities = similarities.mean(axis=1)

# Normalize average similarities to 0-1 range
normalized_similarities = (average_similarities - average_similarities.min()) / (average_similarities.max() - average_similarities.min())

# Map to colors using a colormap
colormap = plt.get_cmap('viridis')
colors = colormap(normalized_similarities)

# Generate HTML with highlighted sentences
html_sentences = []
for i, sentence in enumerate(sentences):
    color = colors[i]  # Color for the current sentence
    color_hex = mcolors.rgb2hex(color) 
    html_sentences.append(f'<span style="background-color: {color_hex};">{sentence.strip()}.</span>')

# Combine into a single HTML document
html_output = "<html><body>" + " ".join(html_sentences) + "</body></html>"

# Save to an HTML file
with open("highlighted_sentences.html", "w", encoding='utf-8') as f:
    f.write(html_output)

print("Highlighted sentences have been saved to highlighted_sentences.html.")