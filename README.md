# 📰 Fake News Analyzer

Fake News Analyzer is a full-stack end-to-end Machine Learning web application designed to predict whether a given news article is real or fake. It utilizes natural language processing (NLP) to analyze article text and provides a confidence percentage along with the final prediction.

## 🚀 Features
- **Accurate Predictions:** Uses an ML pipeline with TF-IDF Vectorization and trained classifying models.
- **Entity Masking:** Integrates `spaCy` to mask named entities to reduce bias towards specific individuals or places and force the model to look at the context.
- **FastAPI Backend:** High-performance, async-ready Python REST API.
- **React Frontend:** Modern, interactive User Interface built with React.js and Vite.
- **Real-time Processing:** Get instant predictions on copied text and titles.

## 🛠️ Technology Stack
- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `spacy` (`en_core_web_sm`)
- **Backend**: FastAPI, Uvicorn, Python 3
- **Frontend**: React (Vite), JavaScript, Vanilla CSS

## 📋 Prerequisites
- Python 3.8+
- Node.js & npm

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/pranavkarvekar/fake-news-analyzer.git
cd fake-news-analyzer
```

### 2. Backend Setup

You can set up the backend using standard `pip` or using `uv` (an extremely fast Python package installer and resolver).

#### Option A: Using standard `pip`
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm
```

#### Option B: Using `uv` (Recommended for speed)
Ensure you have `uv` installed (e.g., `pip install uv`).
```bash
# Create virtual environment
uv venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm
```

To start the FastAPI backend server:
```bash
cd backend
uvicorn app.main:app --reload
```
The backend API will run on `http://localhost:8000`. You can access the API docs at `http://localhost:8000/docs`.

### 3. Frontend Setup
Open a new terminal session.

```bash
cd frontend
npm install
npm run dev
```
The React development server will start on `http://localhost:5173`.

## 🧠 Model Training (Optional)
If you wish to train your own model instead of using the provided pre-trained binaries:
1. Ensure the `WELFake_Dataset.csv` is present in the root folder.
2. Run `python backend/train_model.py`.
3. The newly fine-tuned `pipeline.pkl` will be saved in the `backend/models_bin/` directory.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License
This project is licensed under the MIT License.
