# Atomberg Share of Voice (SoV) Analysis Agent

This project contains a Python-based AI agent created for the Atomberg AI/ML Internship assessment. The agent searches YouTube for "smart fan", analyzes the top 20 results and their comments, and calculates the Share of Voice (SoV) for Atomberg and its competitors.

## Features

* **YouTube Data Mining**: Uses the official YouTube Data API v3 to efficiently fetch video and comment data.
* **Sentiment Analysis**: Employs a pre-trained transformer model from Hugging Face for accurate sentiment scoring.
* **Secure API Key Handling**: Uses a `.env` file to keep API keys safe and out of version control.

## How to Use

### Prerequisites

* Python 3.8+
* A YouTube Data API v3 key from the Google Cloud Console.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kartikde/atomberg-sov-agent.git](https://github.com/kartikde/atomberg-sov-agent.git)
    cd atomberg-sov-agent
    ```

2.  **Create a `.env` file**: In the main project folder, create a file named `.env` and add your API key to it:
    ```
    YOUTUBE_API_KEY="your_actual_api_key_here"
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Script

Execute the script from your terminal:
```bash
python run_analysis.py