from flask import Flask, request, jsonify
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from summarizer import Summarizer
import torch
import wordninja

device = torch.device('cpu')
app = Flask(__name__)

@app.route("/upload", methods=['POST'])
def upload():
    uploaded_file = request.files['file']

    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page in range(num_pages):
        text += pdf_reader.pages[page].extract_text()

    word_split = wordninja.split(text)
    join_text = ' '.join(word_split)
    tokenizer1 = AutoTokenizer.from_pretrained("./models/bart-large")
    model1 = AutoModelForSeq2SeqLM.from_pretrained("./models/bart-large")
    tokenized_text = tokenizer1.encode(join_text, return_tensors='pt').to(device)
    summary_ids = model1.generate(tokenized_text, min_length=40, max_length=120)
    summary = tokenizer1.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({'Summary' : summary})


@app.route('/text', methods=['POST'])
def text_upload():
    json_data = request.get_json()
    data= json_data['text']
    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    #output = summarizer(data, max_length=130, min_length=30, do_sample=False)
    tokenizer1 = AutoTokenizer.from_pretrained("./models/bart-large")
    model1 = AutoModelForSeq2SeqLM.from_pretrained("./models/bart-large")
    tokenized_text = tokenizer1.encode(data, return_tensors='pt').to(device)
    summary_ids = model1.generate(tokenized_text, min_length=40, max_length=120)
    summary = tokenizer1.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({'Summary': summary})


@app.route('/query', methods=['POST'])
def query_upload():
    question_json_data = request.get_json()
    context = question_json_data['context']
    query_json_data = request.get_json()
    query = query_json_data['query']
    tokenizer2 = AutoTokenizer.from_pretrained("./models/rberta")
    model2 = AutoModelForQuestionAnswering.from_pretrained("./models/rberta")
    inputs = tokenizer2(query, context, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model2(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer2.decode(predict_answer_tokens)

    return jsonify({'Answer': answer})
     






if __name__ == '__main__':
    app.run(debug= True)