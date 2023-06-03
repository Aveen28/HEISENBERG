
# IntelliBrief

IntelliBrief is an advanced online application designed to autonomously generate concise summaries for input PDF files or raw textual data. Our innovative platform also incorporates robust query capabilities, facilitating efficient extraction of relevant information from both textual and PDF sources.


## Models used

We have used transformers from [Hugging Face](https://huggingface.co/).
Transformers used in this project are fine-tuned versions of [Bart](https://huggingface.co/facebook/bart-large-cnn) for summarization and [roberta-base](https://huggingface.co/roberta-base) for question answering.

## Features

- Pdf summarization
- Text summarization
- Question answering on PDF and textual data


## Installation

Prerequisites
- [Python](https://www.python.org/downloads/)
- [virtualenv](https://pypi.org/project/virtualenv/)

    
## Run Locally

Clone the project

```bash
  git clone https://github.com/Aveen28/HEISENBERG.git
```

Go to the project directory

```bash
  cd HEISENBERG
```

Setup virtual environment

```bash
  python3 virtualenv env
  source /env/bin/activate
```

Install requirements.txt

```bash
  pip3 install -r requirements.txt
```

Run the project

```bash
  python3 app.py
```
