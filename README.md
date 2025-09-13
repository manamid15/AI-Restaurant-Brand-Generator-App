# AI Restaurant Brand Generator

This is a multi-stage, interactive Streamlit application designed to help users generate a complete brand identity for a new restaurant. It uses a Large Language Model (LLM) from Groq and a RAG (Retrieval-Augmented Generation) system to provide creative and accurate suggestions.

### Features

* **Multi-Stage Generation:** Guides the user from an initial idea to a full brand identity.
* **Name & Tagline Suggestions:** Generates unique restaurant names and catchy taglines based on user input for cuisine and vibe.
* **Advanced Details:** Provides options to generate additional information such as:
    * Suggested dishes
    * Prominent ingredients
    * Top restaurants in the specified cuisine
    * **Intelligent Fallback:** If information is not available in the RAG knowledge base, the application automatically falls back to a Wikipedia search.
* **Static Main Screen:** Once the brand details are finalized, the main content becomes completely static.
* **Interactive Chatbot:** A sidebar chatbot allows users to ask follow-up questions about the chosen cuisine, providing a seamless user experience.
* **Efficient Processing:** Utilizes Groq's fast LLM for quick text generation and FAISS for efficient information retrieval.

### Video Demonstration

A short video demonstration of the application's key features and workflow.

<p align="center">
  <a href="https://www.youtube.com/watch?v=QTJ3UwbWBqg">
    <img src="https://img.youtube.com/vi/QTJ3UwbWBqg/0.jpg" alt="AI Restaurant Brand Generator Video Demo">
  </a>
</p>

---

### How to Run Locally

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/manami15/AI-Restaurant-Brand-Generator.git
    cd AI-Restaurant-Brand-Generator
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all necessary libraries.*

3.  **Set up API key:**
    * Create a Groq account to get your free API key.
    * Create a `.streamlit/secrets.toml` file in your project directory with the following content:
        ```toml
        # .streamlit/secrets.toml
        GROQ_API_KEY="your_api_key_here"
        ```

4.  **Run the app:**
    ```sh
    streamlit run app.py
    ```

### Repository Structure

.<br>
├── .streamlit/<br>
│   └── secrets.toml<br>
├── faiss_index/<br>
├── cuisine_ingredients_ranked.txt<br>
├── .gitignore<br>
├── app.py<br>
├── README.md<br>
└── requirements.txt
