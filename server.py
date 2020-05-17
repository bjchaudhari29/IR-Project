from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
import json
from urllib.parse import parse_qs
from qa import test
import pandas as pd
class S(SimpleHTTPRequestHandler):

    def do_POST(self):
        print( "incomming http: ", self.path )
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode("utf-8") 
        post_data = json.loads(post_data)
        question = post_data['question']
        prod_id = post_data['prod_id']
        
        revi = pd.read_csv('./data/Reviews_125Products.csv')
        n_revi = revi[revi['Product ID'] == int(prod_id)]
        new_rev =''
        for rev in n_revi['Review_Text']:
            rev = rev + '||'
            new_rev = new_rev + rev
        new_rev = new_rev[:-3]
        answers = str(new_rev)
        answers = answers.split('||')
        max_r,question_out,answer_out,found_ques_ans_pair = test(question, answers,int(prod_id))
        if found_ques_ans_pair==0:
            question_out=question_out
            answer_out=answer_out
        else:
            question_out=question_out.values[0]
            answer_out=answer_out.values[0]
        
        response = answers[max_r]
        self.send_response(200) 

        
        self.wfile.write("<html>Review-></br></html>".encode())
        self.wfile.write(response.encode())
        
        if question_out!='empty':
            self.wfile.write("<html><br></br></html>".encode())
            self.wfile.write("<html><head>Question-></br></head></html>".encode())
            self.wfile.write(question_out.encode())
            self.wfile.write("<html></br></html>".encode())
            self.wfile.write("<html><head>Answer-></br></head></html>".encode())
            self.wfile.write(answer_out.encode())

def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print ('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
