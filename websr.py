from flask import Flask, render_template, request, jsonify 
import pandas as pd
from pathlib import Path
import glob
import os
import csv
import nltk
nltk.download('vader_lexicon') #grab it if you need it
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
     
app = Flask(__name__) 
     
@app.route('/') 
def index(): 
    sources = [Path(t).stem.replace('.sentiment','') for t in glob.glob('./data/*.csv')] #["bluets", "foucault"] are default; plus any added by user
    return render_template('index.html', sources=sources) 
     
@app.route('/square/', methods=['POST']) 
def square(): 
    val = str(request.form.get('number', 0))
    # val = str(request.form.get('number'))
    source = str(request.form.get('source', 0))
    print(f'my source is {source}')
    if source == "NewFile":
        fpath = str(request.form.get('filepath', 0))
        fpath = fpath.split('\\')[-1]
        print(fpath)
        get_sentiments(f'./data/{fpath}')
    square = get_log(val, source, 1)
    data = {'square': square} 
    data = jsonify(data) 
    return data 

@app.route('/format/', methods=['POST']) 
def format_text(): 
    # num = float(request.form.get('number', 0)) 
    # square = num ** 2 
    print('lemme format that for ya')
    txt = str(request.form.get('text', 0))
    format_type = str(request.form.get('format', 0))
    new_txt = ''
    if format_type == "prose":
        new_txt = txt.replace('<br>', "  ")
    elif format_type == "log":
        new_txt = txt.replace("  ", "<br>")
    data = {'square': new_txt} 
    data = jsonify(data) 
    return data 

def get_sentiments(file_path):
    filename, file_extension = os.path.splitext(file_path)
    outfile = filename + ".sentiment.csv"
    if not os.path.exists(outfile):
        f = open(file_path, 'r').read()
        sentences = sent_tokenize(f)
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if s != "" and s != '"']

        sentiment = SentimentIntensityAnalyzer()
        sentiments = [] 
        for sentence in sentences:
            compound = sentiment.polarity_scores(sentence)['compound']
            sentiments.append(compound)

        data = sorted((zip(sentences, sentiments)), key = lambda x: x[1])
        fields = ['Line', 'Score']  
        with open(outfile, 'w') as csvout:
            csvwriter = csv.writer(csvout)  
            csvwriter.writerow(fields)
            csvwriter.writerows(data)

    return outfile

def get_log(raw_file, source, curve):
    df = pd.read_csv(f'./data/{source}.sentiment.csv', delimiter=',')
    sentence_data = [list(row) for row in df.values]
    sentiments = [float(x[1]) for x in sentence_data]
    out = ''
    raw_file = raw_file.split('\n')
    for data in raw_file:
        if data:
            sentence_match = biodata_to_text(remap(float(data), 75, 85, -1, 1, curve), sentence_data, sentiments)
            out += sentence_match 
            out += '<br/>'


    return out


def biodata_to_text(data, sentence_data, sentiments):  
    sentiment_match = min(sentiments, key = lambda x: abs(x-data))
    sentence_match = sentence_data[sentiments.index(sentiment_match)]
    print(sentence_match)
    return sentence_match[0]
 
def remap(value, min_i, max_i, min_f, max_f, curve):
    """
    remap a value from on range to another. 
    """
    range_i = max_i - min_i
    range_f = max_f - min_f
    tmp_val = float(value - min_i) / float(range_i)
    new_val = min_f + (tmp_val * range_f)

    if curve == 1:
        return new_val
    elif curve == 2:
        print(new_val**2)
        return new_val**2
    elif curve == 3:
        return new_val**3
    else:
        return new_val

if __name__ == '__main__': 
    	app.run(debug=True) 