# slicing-of-excessive-context-window
## Dynamic Text Processing and Interactive AI Dialogue Management.

# Overview:
This project is designed to demonstrate advanced text preprocessing techniques and interactive dialogue management with a large language model using the Replicate API. The implementation focuses on handling large input sizes that exceed typical model constraints, as well as facilitating an ongoing conversation with an AI model based on dynamically processed text inputs.

# Features:
Text Preprocessing : Cleans and prepares text data by removing HTML tags, non-alphabetic characters, and stopwords, and applying tokenization and lemmatization.
Context Window Slicing : Slices large text inputs into smaller, manageable pieces that fit within the context window of AI models, ensuring efficient processing without losing relevant information.
AI Model Interaction : Utilizes the Replicate API to interact with "meta/llama-2-70b-chat", a large language model, to simulate a realistic dialogue environment where users can ask questions and receive responses.

# Prerequisites:
Before you run the project, make sure you have Python installed on your system along with the following packages:
- BeautifulSoup
- nltk
- scikit-learn
- replicate (You will need an API token from [Replicate](https://replicate.com))

Here’s an enhanced description of your methodology for managing text inputs relative to the context window size of large language models (LLMs), including reference code snippets from your implementation to illustrate key concepts:

 Methodology Overview:

This pipeline is meticulously crafted to optimize interactions with large language models by adeptly managing text input sizes. It is critical for models such as GPT-3 and others with stringent input constraints.

#  Pipeline Description:
1. Handling Inputs Below Context Window Limit
   Direct Processing:
     If the input text size is below the standard size of the context window (128 MB), the text is passed directly to the LLM without modifications. This ensures efficiency in processing texts that naturally fit within the model’s capacity.
     ```python
     # Check if input size is below the context window limit
     if len(input_text.encode('utf-8')) < 128 * 1024 * 1024:
         process_directly(input_text)
     ```

2. Handling Inputs Exceeding Context Window Limit:
   Text Slicing:
     Inputs exceeding the context window are divided into slices that each fit within the 128 MB limit, ensuring that all textual content is processed without exceeding model constraints.
     ```python
     # Function to slice the input text for large inputs
     def generate_slices(input_text, context_window_size=128):
         context_window_bytes = context_window_size * 1024 * 1024
         # Code to slice the text...
     ```

  # Aggregate Size Management:
     The combined size of all slices matches or exceeds the original text length, preserving the integrity and completeness of the input data.
     ```python
     slices = generate_slices(input_text)
     total_length = sum(len(slice.encode('utf-8')) for slice in slices)
     assert total_length >= len(input_text.encode('utf-8'))
     ```

# Criteria for Text Slicing:

Efficient slicing of text inputs is governed by several criteria ensuring each segment's independence and utility:

   Overlap Allowance:
     Overlapping slices maintain contextual continuity, crucial for models heavily reliant on context.
     ```python
     # Example of overlapping slices
     if len(current_slice.encode('utf-8')) + len(word.encode('utf-8')) <= context_window_bytes:
         current_slice += " " + word
     ```

   Exclusion of Redundancy:
     Ensures no slice is fully contained within another, avoiding redundant processing.
     ```python
     # Ensure no complete overlap
     if not is_slice_contained(slices, current_slice):
         slices.append(current_slice.strip())
     ```

   Distinctiveness Requirement:
     Adjacent slices must differ sufficiently, typically enforced via cosine similarity thresholds.
     ```python
     # Check distinctiveness between slices
     vectorizer = TfidfVectorizer()
     tfidf_matrix = vectorizer.fit_transform([slices[i], slices[i+1]])
     cosine_dist = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
     if cosine_dist < 0.2:
         final_slices.append(slices[i+1])
     ```
     Clone the repository, navigate to the project directory, and run the script:
```bash
git clone https://github.com/9158764767/slicing-of-excessive-context-window.git
cd slicing-of-excessive-context-window
pip install -r requirements.txt
python Assignment2_NLP.ipynb



## Acknowledgements
This project was developed as part of the Advanced Text Classification Initiative at University of Verona.
Sincere thanks to all the contributors and maintainers of the  NLTK, scikit-learn, and other open-source projects used in this work.
Special thanks to Professor Prof.Matteo Cristani  for their invaluable guidance and insights throughout the development of this project.

## Contact
For any queries regarding this project, please contact Abhishek Hirve at abhishek.hirve@studenti.univr.it



