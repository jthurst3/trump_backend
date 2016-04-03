
import numpy as np
import nltk
import sys
import re
import random


politicians = ['Donald Trump','Jeb Bush', 'Hillary Clinton', 'Bernie Sanders', 'Barack Obama', 'Jeb Bush', 'Carly Fiorina', 'Marco Rubio', 'Ted Cruz', 'John Kasich', 'Chris Christie', 'Ben Carson']

# extracts specific politicians that have been in the running
def extract_politicians(words):
    # list: Hillary Clinton, Bernie Sanders, Donald Trump, Jeb Bush, Carly Fiorina, Marco Rubio, Ted Cruz, John Kasich, Chris Christie, Ben Carson
    l = []
    for p in politicians:
        fn, ln = p.split()
        for word in words:
            if word == fn or word == ln and p not in l:
                l.append(p)
    return l

# use NLTK tools to extract people's names
def extract_entities(words):
    # inspired by http://timmcnamara.co.nz/post/2650550090/extracting-names-with-6-lines-of-python-code
    l = {'PERSON': [], 'GPE': []}
    for chunk in nltk.ne_chunk(nltk.pos_tag(words)):
        #print chunk
        #print type(chunk)
        if hasattr(chunk, 'label'):
            #print chunk.label(), ' '.join(c[0] for c in chunk.leaves())
            if chunk.label() not in l:
                l[chunk.label()] = []
            j = " ".join(c[0] for c in chunk.leaves())
            if j not in l[chunk.label()]:
                l[chunk.label()].append(j)
    l['GENDERS'] = label_people(l)
    l['MALE'] = [p[0] for p in l['GENDERS'] if p[1] == 'male']
    l['FEMALE'] = [p[0] for p in l['GENDERS'] if p[1] == 'female']
    return l

def label_people(l):
    # distinguish between male and female
    # from http://www.nltk.org/book/ch06.html
    labeled_names = {}
    for name in nltk.corpus.names.words('male.txt'):
        labeled_names[name] = 'male'
    for name in nltk.corpus.names.words('female.txt'):
        labeled_names[name] = 'female'
    if 'PERSON' not in l:
        return []
    else:
        classified_people = []
        for p in l['PERSON']:
            if p in labeled_names and len([p1 for p1 in classified_people if p1[0] == p]) == 0:
                classified_people.append((p, labeled_names[p]))
            else:
                sp = p.split()
                found = False
                for p1 in sp:
                    if p1 in labeled_names and len([p2 for p2 in classified_people if p2[0] == p]) == 0:
                        classified_people.append((p, labeled_names[p1]))
                        found = True
                # assume names we don't know are female (just for fun)
                if not found:
                    classified_people.append((p, 'female'))
        return classified_people

def extract_things(words):
    l = []
    for t in nltk.pos_tag(words):
        if type(t) == tuple and len(t) == 2 and t[1] == "NN" and t[0] not in l:
            l.append(t[0])
    return l

def extract_keywords(words):
    keyword_list = ['China','Mexico','North Korea', 'women', 'fat', 'slob', 'Muslim', 'wall', 'win']
    l = []
    for k in keyword_list:
        for word in words:
            if word.lower() == k.lower() and word.lower() not in l:
                l.append(word.lower())
    return l

def extract_type_of_question(words):
    question_words = ['who', 'what', 'where', 'when', 'why', 'how']
    if words[0].lower() in question_words:
        return words[0]
    return 'unknown'

# format for input data:
#{
        #'politicians': (politicians mentioned)
        #'people': (people mentioned)
        #'things': (names of things mentioned (e.g. hackathons))
        #'keywords': (specific things like women that we want to sort of hard code)
        #'type': (type of question, e.g. who, what, why, how, ...)
#}
def parse_input():
    words = nltk.word_tokenize(raw_input())
    # extract politicians
    d = {}
    d['POLITICIAN'] = extract_politicians(words)
    d['entities'] = extract_entities(words)
    d['PERSON'] = d['entities']['PERSON']
    d['MALE'] = d['entities']['MALE']
    d['FEMALE'] = d['entities']['FEMALE']
    d['GPE'] = d['entities']['GPE']
    d['THING'] = extract_things(words)
    d['keywords'] = extract_keywords(words)
    d['type'] = extract_type_of_question(words)
    return d

# parse the trumpisms text file
def parse_trumpisms():
    t = {'prefixes': [], 'suffixes': [], 'phrases': [], 'standalones': []}
    with open('trumpisms.txt') as f:
        for line in f:
            sp = line.split()
            j = ' '.join(sp[1:])
            if sp[0] == "PREFIX":
                t['prefixes'].append(j)
            elif sp[0] == "SUFFIX":
                t['suffixes'].append(j)
            elif sp[0] == "PHRASE":
                t['phrases'].append(j)
            elif sp[0] == "STANDALONE":
                t['standalones'].append(j)
    return t

def add_prefix(trumpisms):
    if random.randint(0,4) == 0:
        t = trumpisms['prefixes']
        random.shuffle(t)
        return t[0]
    else:
        return ''

def add_suffix(trumpisms):
    if random.randint(0,4) == 0:
        t = trumpisms['suffixes']
        random.shuffle(t)
        return t[0]
    else:
        return ''

def answer(input_data, trumpisms):
    # determine which trumpisms we can use for each thing
    usable_sentences = {}
    r1 = re.compile('\$\{[0-9]+\:[A-Z,]+\}')
    r2 = re.compile('\(.*?\)')
    categories = ['PERSON','GPE','POLITICIAN', 'THING', 'MALE', 'FEMALE']
    for c in categories:
        for t in trumpisms['phrases']:
            matches = set(re.findall(r1, t))
            for elem in input_data[c]:
                sub = r1.sub(elem, t)
                options = re.findall(r2,t)
                sub2s = [r2.sub('',t)]
                for o in options:
                    text = r2.sub(elem, o)[1:-1]
                    s = r2.sub(text,t)
                    sub2s.append(s)
                if elem not in usable_sentences:
                    usable_sentences[elem] = []
                for m in matches:
                    if c in m and sub not in usable_sentences[elem]:
                        usable_sentences[elem].append(sub)
    topics = []
    if len(usable_sentences.keys()) == 0:
        return ["You know, I don't understand that question. That's a stupid question, that's a really stupid question."]
    elif len(usable_sentences.keys()) == 1:
        topics.append(usable_sentences[usable_sentences.keys()[0]])
    else:
        k = usable_sentences.keys()
        random.shuffle(k)
        for i in range(2):
            topics.append(usable_sentences[k[i]])
    sentences = []
    for t in topics:
        random.shuffle(t)
        for i in range(min(len(t), random.randint(3,5))):
            pref = add_prefix(trumpisms)
            suff = add_suffix(trumpisms)
            s = ""
            if pref:
                s += pref + ", "
            s += t[i]
            if suff:
                s += ", " + suff
            s += "."
            sentences.append(s)

    return sentences

def main():
    trumpisms = parse_trumpisms()
    input_data = parse_input()
    sentences = answer(input_data, trumpisms)
    print ' '.join(sentences)


if __name__ == '__main__':
    main()
