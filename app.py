# Serve model as a flask application

import torch
from flask import Flask, request, render_template
from CryptoSentimentAnalysis import models
from transformers import DistilBertTokenizer, DistilBertModel
model = None
app = Flask(__name__)

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        # self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def load_model():
    global model
    global tokenizer
    # model variable refers to the global variable
    model_dir = './CryptoSentimentAnalysis/SavedModels/'
    path = model_dir + "pytorch_distilbert_sent_uncased.pth"

    device = torch.device('cpu')

    model = DistillBERTClass()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    #model = models.DistillBERTClass()
    #model = DistilBertModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    #model = torch.load(path)
    #model.to(device)
    #model.eval()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Works only for a single sample
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        prediction = models.Tweet_Predict(data[0], 0, tokenizer, model)  # runs globally loaded model on the data
    return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)