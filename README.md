# Question Augmentation Tool

This tool automatically generates question-answer pairs from text documents using the Hugging Face Inference API. It processes text documents, splits them into manageable chunks, and uses a language model to generate relevant Q&A pairs based on the content.

## Features

- Process single documents or entire directories
- Automatic chunking of large documents
- Configurable number of Q&A pairs per chunk
- Retry mechanism for API failures
- Intermediate result saving to prevent progress loss
- Filtering of placeholder/template entries

## Setup

### Prerequisites

- Python 3.6+
- A Hugging Face account with an API token ([Get a token here](https://huggingface.co/settings/tokens))

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/mohitvermax/question-answer-aggregator
   cd question-answer-aggregator
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory with your Hugging Face API token:
   ```
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

## Usage

### Basic Usage

To generate Q&A pairs from a single document:

```bash
python question_augmentation.py --input path/to/document.txt --output qa_dataset.json
```

To process all text files in a directory:

```bash
python question_augmentation.py --input path/to/documents/ --output qa_dataset.json
```

### Command-line Options

- `--model`: Hugging Face model name (default: mistralai/Mistral-7B-Instruct-v0.2)
- `--input`: Path to input document or directory (required)
- `--output`: Output JSON file path (default: qa_dataset.json)
- `--pairs-per-chunk`: Number of Q&A pairs to generate per chunk (default: 3)
- `--retries`: Number of retries for failed API calls (default: 3)

### Example

```bash
python question_augmentation.py --model meta-llama/Llama-2-7b-chat-hf --input documents/ --output my_dataset.json --pairs-per-chunk 5
```

## Output Format

The tool generates a JSON file with the following structure:

```json
[
  {
    "question": "What is the main topic of this document?",
    "answer": "The document discusses the principles of machine learning."
  },
  {
    "question": "How is supervised learning defined in the text?",
    "answer": "Supervised learning is defined as a process where the model learns from labeled examples."
  },
  ...
]
```

## Troubleshooting

- **Permission Errors**: Make sure your Hugging Face token has the appropriate permissions and the model is accessible.
- **Slow Generation**: Some larger models may take time to load. The tool implements a waiting mechanism for this.
- **JSON Parsing Errors**: The tool includes robust error handling for various formats returned by different models.

## Recommended Models

If you encounter permission issues, try these publicly available models:

- mistralai/Mistral-7B-Instruct-v0.2
- microsoft/Phi-2
- tiiuae/falcon-7b-instruct
- EleutherAI/gpt-neox-20b
- bigscience/bloom-7b1
