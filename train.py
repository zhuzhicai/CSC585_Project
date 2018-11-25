import json
import torch
import numpy as np
import re
import spacy
from collections import Counter
import sys
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt

# Flatten the json file
def get_example(data):
    count = 0
    examples = []
    example = {'answer':[], 'answer_start':[], 'answer_length':[]}
    for topic in data:
        #print topic['title']
        example['title'] = topic['title']
        for paragraph in topic['paragraphs']:
            example['context'] = paragraph['context']
            for qa in paragraph['qas']:
                example['question'] = qa['question']
                example['is_impossible'] = qa['is_impossible']
                example['question_id'] = qa['id']
                for answer in qa['answers']:
                    example['answer'].append(answer['text'])
                    example['answer_start'].append(answer['answer_start'])
                    example['answer_length'].append(len(answer['text']))
                examples.append(example.copy())
                example['answer'] = []
                example['answer_start'] = []
                example['answer_length'] = []
    return examples

# Convert to word2vec dictionary
def read_in_w2v(filename):
#     word2vec_counter = open('../counter-fitted-vectors.txt', 'r')
    word2vec_dict = open(filename, 'r')
    vectors = word2vec_dict.readlines()

    w2v_dict = {}

    for vector in vectors:
        v = vector.split()
        w2v_dict[v[0]] = torch.tensor([float(i) for i in v[1:]])
        w2v_dict[v[0]] = w2v_dict[v[0]].unsqueeze(0)
    
    return w2v_dict

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def word_tokenizer_pos_reg_indexes(text):
    tokens = []
    pos_tags = []
    ner_tags = []
    
    nlp = spacy.load('en')
    doc = nlp(text)
    
    for token in doc:
        flag = 0
        tokens.append(token.text)
        pos_tags.append(token.tag_)
        for ent in doc.ents:
            if token.text in ent.text:
                ner_tags.append(ent.label_)
                flag = 1
        if flag == 0:
            ner_tags.append('OUT')
            
    token_indexes = convert_idx(text, tokens)
        
    return tokens, pos_tags, ner_tags, token_indexes

def sentences2mat(text, w2v_dict, pos_dict, ner_dict, unknown_vec):
    text = text.replace("\"", "``").replace("\"","''")
    known_set = set()
    unknown_set = set()
    tokens, pos_tags, ner_tags, token_indexes = word_tokenizer_pos_reg_indexes(text)

    for index,token in enumerate(tokens):
        token = token.lower()
        if token in w2v_dict:
            known_set.add(token)
            word_tensor = w2v_dict[token]
        else:
            unknown_set.add(token)
            word_tensor = unknown_vec

        word_tensor = torch.cat([word_tensor, torch.tensor([[pos_dict[pos_tags[index]], ner_dict[ner_tags[index]]]])],1)
        
        if index == 0:
            sentense_mat = word_tensor
        else:
            sentense_mat = torch.cat((sentense_mat, word_tensor),0)
    
    return sentense_mat, known_set, unknown_set, token_indexes

# Preprocess the given data to word_embedding+pos+ner
def prepro(filename1, filename2):
    data = json.load(open(filename1, 'r'))
    x = get_example(data['data'])
    w2v_cf_dict = read_in_w2v(filename2)

    known_word_set = set()
    unknown_word_set = set()
    unrecognized_bow = {}
    context2mat = {}
    question2mat = {}
    context2indices = {}
    question2indices = {}
    answer_vector = {}

    # some rules to mapping words:
    pos_list = ['','-LRB-','-RRB-',',', ':', '.', '\'\'', '\"\"', '#', '``', '$', 'ADD', 'AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'FW','GW','HVS','HYPH','IN','JJ','JJR','JJS','LS','MD','NFP','NIL','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','_SP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','XX']
    ner_list = ['OUT','PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
    unknown_vec = torch.zeros(1,300)

    pos_dict,ner_dict = {},{}
    for index,pos_tag in enumerate(pos_list):
        pos_dict[pos_tag] = (index)/100.0
    for index,ner_tag in enumerate(ner_list):
        ner_dict[ner_tag] = (index)/100.0

    # convert all examples to matrixes
    # get the length of tokens of contexts and questions
    context_length = []
    question_length = []
    counter = 0

    for item in x:
        counter += 1
        if counter %1e4 == 0:
            print(counter)
        if item['question_id'] not in unrecognized_bow:
            unrecognized_bow = {item['question_id']:{'context':set(), 'question':set()}}
        if item['context'] not in context2mat:
            context2mat[item['context']], known_set, unknown_set, context2indices['token_indexes'] = sentences2mat(item['context'],w2v_cf_dict, pos_dict, ner_dict, unknown_vec)
            known_word_set.update(known_set)
            unknown_word_set.update(unknown_set)
            unrecognized_bow[item['question_id']]['context'] = unknown_set
            context_length.append(len(context2indices['token_indexes']))        
        if item['question'] not in question2mat:
            question2mat[item['question']], known_set, unknown_set, question2indices['token_indexes'] = sentences2mat(item['question'],w2v_cf_dict, pos_dict, ner_dict, unknown_vec)
            known_word_set.update(known_set)
            unknown_word_set.update(unknown_set)
            unrecognized_bow[item['question_id']]['question'] = unknown_set
            question_length.append(len(question2indices['token_indexes']))
        # generate answer vector: one-hot vector
        if item['is_impossible'] == True:
            answer_vector[item['question_id']] = torch.zeros(len(context2indices['token_indexes']))
        else: 
            answer_vector[item['question_id']] = torch.zeros(len(context2indices['token_indexes']))
            for k,v in enumerate(context2indices['token_indexes']):
                if v[0] >= item['answer_start'][0] and v[1] <= item['answer_start'][0]+item['answer_length'][0]:
                    answer_vector[item['question_id']][k] = 1      
    print("Totally, we have {} out of {} words got vector".format(len(known_word_set), len(unknown_word_set)+len(known_word_set)))                    
    print(unknown_word_set)  

    example_dict = {}
    for item in x:
        if item['question_id'] not in example_dict:
            example_dict[item['question_id']] = {}
            example_dict[item['question_id']]['context']=context2mat[item['context']]
            example_dict[item['question_id']]['question']=question2mat[item['question']]
            example_dict[item['question_id']]['answer']=answer_vector[item['question_id']]
    return example_dict

# define the model we want to use
class SimpleQA(nn.Module):
    def __init__(self):
        super(SimpleQA, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=302, out_channels=300, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=302, out_channels=300, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels=300, out_channels=200, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=300, out_channels=200, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=200, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv6 = nn.Conv1d(in_channels=200, out_channels=50, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv7 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv8 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv9 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv10 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv11 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv12 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,50)
#         self.maxPool = nn.MaxPool1d(50)
#         self.line2 = nn.Linear(50,1)
    def forward(self, c, q):
        f1 = self.conv1(c)                                
        f1 = torch.relu(f1)
        f1 = self.conv3(f1)
        f1 = torch.relu(f1)
        f1 = self.conv5(f1)
        f1 = torch.relu(f1)
        f1 = self.conv7(f1)
        f1 = torch.relu(f1)
        f1 = self.conv8(f1)
        f1 = torch.relu(f1)
        f1 = self.conv9(f1)
        f1 = torch.relu(f1)
        f1 = self.conv10(f1)
        f1 = torch.relu(f1)        
        f1 = self.conv11(f1)
        f1 = torch.relu(f1) 
        f1 = self.conv12(f1)
        f1 = torch.relu(f1)
        # f1-> c_word_count * 20
#         plt.imshow(f1[0].detach().numpy())
#         plt.show()
                
        f2 = self.conv2(q)
        f2 = torch.relu(f2)
        f2 = self.conv4(f2)
        f2 = torch.relu(f2)
        f2 = self.conv6(f2)
        f2 = torch.relu(f2)
        f2 = f2.max(dim=2)[0]
        f2 = f2.softmax(dim=0)
#         f2 = self.maxPool(f2.transpose(1,2))
        #f2-> q_word_count * 50
        
        f1 = self.fc1(f1.transpose(1,2))
        f1 = f1[0]
        f1 = torch.relu(f1)
        f1 = self.fc2(f1)
        f1 = torch.mm(f1, f2.transpose(0,1))
        f1 = torch.relu(f1)
        f1 = f1.softmax(dim=0)
        return f1.transpose(0,1)[0]

def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    example_dict = prepro(filename1, filename2)

    simpleqa = SimpleQA()
    optimizer = optim.SGD(simpleqa.parameters(), lr=1e-2, momentum=0.5)
    lossFunction = nn.MSELoss()

    torch.cuda.is_available()


    lossHistory = []
    for epoch in range(1000):

        if epoch+1 % 10 == 0:
            torch.save(simpleqa.state_dict(), str(epoch)+'.model')

        for i, (k,v) in enumerate(example_dict.items()):        
            c = v['context']
            c = c.transpose(0, 1)
            c = c.unsqueeze(0)
            q = v['question']
            q = q.transpose(0, 1)
            q = q.unsqueeze(0)
    #         y = simpleqa.conv2(q)
    #         y = torch.relu(y)

    #         f1 = simpleqa.conv1(c)
    #         f1 = torch.relu(f1)
            
    #         f = simpleqa.line1(f1.transpose(1,2))
    #         print(f.shape)

            yPred = simpleqa(c, q)
    #         print(yPred)
            yPred = yPred/max(yPred)
    #         print(yPred)
            
    #         print(v['answer'])
    #         print(yPred.shape)
            target = v['answer']
            if epoch >50:
                print(target)
                print(yPred)
                print('/n')
            loss = lossFunction(yPred, target)
            lossHistory.append(loss.item())

            simpleqa.zero_grad()
            loss.backward()
            optimizer.step()
            
            
    #         plt.plot(simpleqa(c,q).detach().numpy())
    #         plt.plot(v['answer'].numpy())
    #         plt.show()
            
            
            if i%5000 == 4999:
                plt.plot(lossHistory, '.-')
                plt.save('epoch'+str(epoch)+'_'+str(i)+'.png')




if __name__ == "__main__":
    main()



