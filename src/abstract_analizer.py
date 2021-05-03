import stanza
from stanza_batch import batch
from types import SimpleNamespace
from .default_preprocessing_modules import create_default_nlp_module

def flat_article(article):
    sentence_list = []
    sentence_paragraph = []
    prev_end = 0
    for i, para in enumerate(article):
        sentence_list.extend(para)
        sentence_paragraph.append(prev_end+len(para))
        prev_end=prev_end+len(para)
    sentence_list = [s if s.strip() else "-" for s in sentence_list]
    return sentence_list, sentence_paragraph

def unflat_article(sentence_list, sentence_paragraph):
    article = []
    for i in range(len(sentence_paragraph)):
        start = 0
        end = 0
        if i!=0:
            start = sentence_paragraph[i-1]
        end = sentence_paragraph[i]
        article.append([s if s!="-" else "" for s in sentence_list[start:end]])
    return article

class AbstractAnalizer:
    def __init__(self, language='en', nlp_module=None):
        # load the stanza model for tokenizing
        if nlp_module is None:
            self.nlp = create_default_nlp_module(language)
        else:
            self.nlp = nlp_module

    def load_document(self, input, language):
        if(isinstance(input,str)):
            paragraphs = input.split('\n')
            tokenized_paragraphs = [self.nlp(p) for p in paragraphs if p.strip()]
        elif isinstance(input,list):
            paragraphs = [' '.join(para) for para in input]
            sentence_list, sentence_paragraph = flat_article(input)
            sentence_nlp = [doc.sentences[0] for doc in batch(sentence_list, self.nlp,batch_size=32)]
            article_rebuilt = unflat_article(sentence_nlp, sentence_paragraph)
            tokenized_paragraphs = [SimpleNamespace(sentences=para) for para in article_rebuilt]
            # tokenized_paragraphs = [SimpleNamespace(sentences=[self.nlp(sentence).sentences[0] for sentence in para if sentence.strip()]) for para in input]
            # tokenized_paragraphs = [SimpleNamespace(sentences=[doc.sentences[0] for doc in batch(para, self.nlp,batch_size=32)]) for para in input]
        else:
            raise NotImplementedError()
        # print(paragraphs)

        # tokenizing, lemmarization, ...

        return {
            'paragraphs': paragraphs,
            'tokenized_paragraphs': tokenized_paragraphs,
            'language': language
        }