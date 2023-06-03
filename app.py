from flask import Flask, render_template,request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import torch
import os
import textract
global que

app = Flask(__name__)

device = torch.device('cpu')

@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/index_2')
def index_2():
    return render_template('index2.html')

@app.route('/index_3')
def index_3():
    return render_template('index3.html')

@app.route('/index_4')
def index_4():
    return render_template('index4.html')

@app.route('/index_5')
def index_5():
    return render_template('index5.html')

@app.route('/index_1')
def index_1():
    return render_template('index1.html')

@app.route('/display', methods=['POST'])
def display():
    data = request.form['variable']
    tokenizer1 = AutoTokenizer.from_pretrained("models/bart")
    model1 = AutoModelForSeq2SeqLM.from_pretrained("models/bart")
    tokenized_text = tokenizer1.encode(data, return_tensors='pt').to(device)
    summary_ids = model1.generate(tokenized_text, min_length=40, max_length=120)
    summary = tokenizer1.decode(summary_ids[0], skip_special_tokens=True)

    return render_template('display.html', data=summary)






@app.route('/query', methods=['POST'])
def query():
    variable1 = request.form['variable1']
    variable2 = request.form['variable2']

    tokenizer2 = AutoTokenizer.from_pretrained("models/rbert")
    model2 = AutoModelForQuestionAnswering.from_pretrained("models/rbert")
    inputs = tokenizer2(variable2, variable1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model2(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer2.decode(predict_answer_tokens)
    print(type(answer))




    return render_template('display1.html', variable1=answer)
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return 'No file selected'
    

    # Check if the selected file is a PDF
    if file.mimetype != 'application/pdf':
        return 'Please upload a PDF file'

    # Save the file to a folder
    file.save(os.path.join(os.getcwd(), file.filename))

    # Get the file path on the server
    file_path = os.path.join(os.getcwd(), file.filename)

    byte_data = textract.process(file_path)

    text_data = byte_data.decode()

    tokenizer1 = AutoTokenizer.from_pretrained("models/bart")
    model1 = AutoModelForSeq2SeqLM.from_pretrained("models/bart")
    tokenized_text = tokenizer1.encode(text_data, return_tensors='pt').to(device)
    summary_ids = model1.generate(tokenized_text, min_length=40, max_length=120)
    summary = tokenizer1.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)



    return render_template('index3.html',variable1=summary)

@app.route('/displayss', methods=['POST'])
def displayss():
    query = request.form['variable2']
    text_data = textract.process('pdffff-1.pdf').decode()
    tokenizer1 = AutoTokenizer.from_pretrained("models/rbert")
    model1 = AutoModelForQuestionAnswering.from_pretrained("models/rbert")
    inputs = tokenizer1(query, text_data, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model1(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    data = tokenizer1.decode(predict_answer_tokens)
    



    return render_template('index5.html', data=data)




if __name__ == '__main__':
    app.run()


