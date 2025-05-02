# Java Project Evaluator

A web application for automated evaluation of Java student projects against dynamically extracted assessment criteria from a provided brief, powered by local LLMs (Ollama + Gemma), LangChain, and PostgreSQL.

---

## Features

- **Upload Java Projects:** Upload a ZIP file containing multiple student Java projects.
- **Assessment Brief Extraction:** Upload a PDF assessment brief; criteria are extracted using a local LLM (Gemma via Ollama).
- **Customizable LLM Prompt:** Tweak the preamble and postamble for criteria extraction directly from the web UI.
- **Automated Evaluation:** Each student project is compared against the extracted criteria using Retrieval-Augmented Generation (RAG).
- **Results Dashboard:** View evaluation results in the web interface.
- **Robust Storage:** All files and results are stored securely; PostgreSQL is used for persistent data.
- **Modern Stack:** Flask, Flask-SocketIO, LangChain, ChromaDB, HuggingFace embeddings, Docker, and Ollama.

---

## Getting Started

### Prerequisites

- [Python 3.9+](https://www.python.org/)
- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/) (for local LLM inference)
- [Git](https://git-scm.com/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/java-eval.git
   cd java-eval
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in the required values.

4. **Start PostgreSQL with Docker:**
   ```bash
   docker-compose up -d
   ```

5. **Start Ollama and pull the Gemma model:**
   ```bash
   ollama run gemma
   ```

6. **Run the Flask app:**
   ```bash
   flask run
   ```
   Or, if using SocketIO:
   ```bash
   python app.py
   ```

---

## Usage

1. **Open the web interface:**  
   Go to [http://localhost:5000](http://localhost:5000) in your browser.

2. **Upload the assessment brief PDF.**

3. **Customize the LLM prompt (optional):**  
   Edit the preamble and postamble fields to guide criteria extraction.

4. **Upload a ZIP of student Java projects.**

5. **View evaluation results and download reports.**

---

## Project Structure 
.
├── app.py
├── requirements.txt
├── docker-compose.yml
├── storage/
│ ├── assessment_brief/
│ └── submissions/
├── templates/
├── static/
├── db_utils.py
├── .env.example
└── ...

---

## Technologies Used

- **Flask** (web framework)
- **PostgreSQL** (database, via Docker)
- **LangChain** (RAG, LLM orchestration)
- **Ollama** (local LLM inference, Gemma model)
- **ChromaDB** (vector storage)
- **HuggingFace Embeddings**
- **Docker** (containerization)

---

## Security & Notes

- **Do not commit your `.env` file or any sensitive data.**
- All LLM calls are local (no data leaves your machine).
- Uploaded files are stored in the `storage/` directory (see `.gitignore`).

---

## License

[MIT](LICENSE)

---

## Acknowledgments

- [Ollama](https://ollama.com/)
- [LangChain](https://www.langchain.com/)
- [Flask](https://flask.palletsprojects.com/)
- [ChromaDB](https://www.trychroma.com/)
