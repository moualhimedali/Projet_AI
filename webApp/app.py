from flask import Flask, render_template, request, jsonify
from simplet5 import SimpleT5
import ampligraph
from ampligraph.utils import restore_model
from ampligraph.discovery import query_topn, find_clusters
import pandas as pd

query2triplet_model = SimpleT5()
query2triplet_model.load_model("t5","models/query2triplet_model", use_gpu=False)


knowledge_graph_model = restore_model('distMult.pkl')

from ctransformers import AutoModelForCausalLM
llm = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GGUF", model_file="zephyr-7b-beta.Q4_K_M.gguf", model_type="mistral")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    
    triplet = query2triplet_model.predict(text)
    head, relation, tail = tuple(triplet[0].split(','))
    if head == '?':
        head = None
    elif relation == '?':
        relation = None
    elif tail == '?':
        tail = None

    """
    triplets, scores = query_topn(knowledge_graph_model, top_n=100, 
                                 head=head, 
                                 relation=None, 
                                 tail=head, 
                                 ents_to_consider=None, 
                                 rels_to_consider=None)
    """
    question = text
    dataset = pd.read_csv('triplets.csv', index_col=0)
    dataset.fillna('', inplace=True)
    triplets = dataset[
        (dataset['subject'] == head) | (dataset['object'] == head) | (dataset['subject'] == tail) | (dataset['object'] == tail)
        ].values
    context = ''
    for triplet in triplets:
        context = context + ' '.join(triplet)
    
    if len(context.split(' ')) > 128:
        context_tokens = context.split(' ')
        context = ' '.join(context_tokens[:128])
    
    QA_input = 'context: ' + context + '. answer to the question using the context: ' + question
    
    QA_output = llm(QA_input, max_new_tokens=128)
        
    return QA_output


if __name__ == '__main__':
    app.run()
