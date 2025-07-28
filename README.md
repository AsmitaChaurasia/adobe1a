Adobe Hackathon: Connecting the Dots (Round 1 Submission)
This project is a solution for Round 1 of the "Connecting the Dots" Adobe India Hackathon. It implements a two-stage pipeline to first understand the structure of PDF documents and then use that structure to perform persona-driven document intelligence.
Approach
Our solution is a robust pipeline that intelligently parses, retrieves, and ranks information from a collection of PDF documents based on a user's specific role and task.
Stage 1: ML-Powered Document Structure Analysis (Round 1A)
The foundation of our system is a custom-trained machine learning model that accurately identifies the structure of a PDF.
Feature Extraction: We use the PyMuPDF library to extract rich features for every line of text in a document, including font size, boldness, word count, vertical position on the page, and numbering patterns.

Model Training: A RandomForestClassifier is trained on a labeled dataset to classify each line as a Title, H1, H2, H3, or Body Text. This model, heading_classifier_gold.joblib, is the "brain" that understands document hierarchy.

Stage 2: Persona-Driven Intelligence Pipeline (Round 1B)
This stage uses the structural understanding from Stage 1 to perform a deep, semantic analysis of the document collection.

Parse: The system first uses the trained ML model to parse all input PDFs. It groups all body text under its corresponding heading, transforming the documents into a list of structured, meaningful text "chunks."

Retrieve: We employ a state-of-the-art semantic search system.

Embeddings: The sentence-transformers library (all-MiniLM-L6-v2 model) is used to create contextual vector embeddings for every text chunk. A query embedding is also created from the user's persona and job_to_be_done.

Search: A FAISS (Facebook AI Similarity Search) index is built from the document embeddings. This allows for an incredibly fast and efficient semantic search to find the chunks that are most conceptually similar to the user's query.

Rank & Analyze: The retrieved chunks, each with a similarity score, are intelligently aggregated. The scores for all chunks belonging to the same document section are summed up. This allows us to rank the sections by their overall relevance. The top-ranked sections and their most relevant text are then formatted into the final JSON output.

Models & Libraries
scikit-learn: For training the RandomForestClassifier.

sentence-transformers: For generating high-quality semantic embeddings.

faiss-cpu: For building a high-speed vector search index.

PyMuPDF: For robust PDF parsing and feature extraction.

torch: As a dependency for sentence-transformers.

pandas & numpy: For efficient data manipulation.

How to Build and Run
This solution is packaged in a Docker container for easy and reliable execution.
#  Build the Docker Image
Navigate to the adobe1b directory in your terminal and run the build command:
Bash
docker build -t mysolution .
# Run the Container
To run the solution, you must mount a local input directory to the container's /app/input and a local output directory to /app/output.
From the parent adobechallengehackathon directory, run the following command. This example uses the input1 test case.
Bash
# For Mac/Linux:
docker run --rm -v "$(pwd)/adobe1b/input1:/app/input" -v "$(pwd)/adobe1b/outputs:/app/output" mysolution

# For Windows (PowerShell):
docker run --rm -v "${pwd}/adobe1b/input1:/app/input" -v "${pwd}/adobe1b/outputs:/app/output" mysolution
The application will automatically find the input files, process them, and save the challenge1b_ml_output.json file to your local adobe1b/outputs folder.