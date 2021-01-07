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
THE_DATA = []

@app.route('/') 
def index(): 
    sources = [Path(t).stem.replace('.sentiment','') for t in glob.glob('./data/*.csv')] #["bluets", "foucault"] are default; plus any added by user
    print(f"my source materials are:{sources}")
    return render_template('index.html', sources=sources) 

@app.route('/post/', methods = ["POST"])
def post():
    print(request.get_data().decode())
    return 'sup'
     
@app.route('/run/', methods=['POST']) 
def run(): 
    val = str(request.form.get('number', 0))
    source = str(request.form.get('source', 0))
    txt_format = str(request.form.get('format', 0))
    curve = int(request.form.get('curve', 1))
    print(f'my source is {source}')
    if source == "NewFile":
        fpath = str(request.form.get('filepath', 0))
        fpath = fpath.split('\\')[-1]
        print(fpath)
        get_sentiments(f'./data/{fpath}')
        return jsonify({'text': f'got the sentiments from {fpath}'})
    txt = get_log(val, source, curve, txt_format)
    data = {'text': txt} 
    data = jsonify(data) 
    return data 

@app.route('/realtime/', methods=['POST']) 
def realtime(source, curve):
    outfile = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + f"_{source}.txt"
    raw_outfile = "./logs/raw/" + outfile
    text_outfile = "./logs/text/" + outfile
    f1 = open(raw_outfile, 'a')
    f2 = open(text_outfile, 'a')
    df = pd.read_csv(f'./data/{source}.sentiment.csv', delimiter=',')
    sentence_data = [list(row) for row in df.values]
    sentiments = [float(x[1]) for x in sentence_data]
    while True:
        try:
            ser_bytes = ser.readline()
            decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
            data = remap(decoded_bytes, 75, 85, -1, 1, curve) #input range will vary on biodata(/person/etc). output should be [-1,1] for compound sentiment 
            sentence = biodata_to_text(data, sentence_data, sentiments)
            f1.write(str(decoded_bytes)+'\n')
            f2.write(sentence+'\n\n')
        except KeyboardInterrupt:
            f1.close()
            f2.close()
            print(f'data saved to {raw_outfile} and {text_outfile}')
            exit(0)

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
    data = {'text': new_txt} 
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

def get_log(raw_file, source, curve, txt_format):
    df = pd.read_csv(f'./data/{source}.sentiment.csv', delimiter=',')
    sentence_data = [list(row) for row in df.values]
    sentiments = [float(x[1]) for x in sentence_data]
    out = ''
    raw_file = raw_file.split('\n')
    for data in raw_file:
        if data:
            sentence_match = biodata_to_text(remap(float(data), 75, 85, -1, 1, curve), sentence_data, sentiments)
            out += sentence_match 
            if txt_format == 'log':
                out += '<br>'
            elif txt_format == 'prose':
                out += '  '
    return out


def biodata_to_text(data, sentence_data, sentiments):  
    sentiment_match = min(sentiments, key = lambda x: abs(x-data))
    sentence_match = sentence_data[sentiments.index(sentiment_match)]
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
    	app.run(host='0.0.0.0', debug=True) 