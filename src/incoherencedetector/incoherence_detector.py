# -*- coding: utf-8 -*-
import stanza
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertModel
import json
import numpy as np
from itertools import product
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path

from analizers.default_preprocessing_modules import create_default_nlp_module

from analizers.abstract_analizer import AbstractAnalizer
from .custom_dataset import CustomDataset
from .next_sentence_classifier import NextSentenceClassifier


class IncoherenceDetector(AbstractAnalizer):
    def __init__(self, language='en', nlp_module=None):
        super().__init__(language, nlp_module)

        self.use_cuda = torch.cuda.is_available()
        print("Keyword extractor using GPU: "+str(self.use_cuda))
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # self.device = torch.device('cpu')

        base_path = Path(__file__).parent
        checkpoint_filename = (
            base_path / "./checkpoint_mixed_bert-base-multilingual-cased_pooler_0_768.pt").resolve()
        checkpoint = torch.load(checkpoint_filename, map_location='cpu')

        # 1. filter out unnecessary keys
        # pretrained_dict = {
        #     k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('fc2')}
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint['bert_model_name'])

        self.model = NextSentenceClassifier(
            bert_model_name=checkpoint['bert_model_name'],
            state_dict=checkpoint['model_state_dict']
        )

        # quantize the encoder model (BERT) to int8 on cpu
        if not self.use_cuda:
            self.model.bert_layer = torch.quantization.quantize_dynamic(self.model.bert_layer,{torch.nn.Linear},dtype=torch.qint8, inplace=True)

        self.model.to(self.device)

        self.model.eval()

    def execute(self, state):
        paragraphs = [[sentence.text for sentence in paragraph.sentences]
                      for paragraph in state['tokenized_paragraphs']]
        result = []
        # print(paragraphs)
        # process with model
        for paragraph in paragraphs:
            if len(paragraph) > 2:
                masked_arr, cell_list = self.create_mask_matrix(
                    len(paragraph), len(paragraph), 3)
                matrix_rounded, matrix_original = self.create_paragraph_matrix(
                    self.model, paragraph, mask=masked_arr)

                _, has_missing, _, missing_sentences = self.process_paragraph(
                    self.model, paragraph, 1, cmatrix=matrix_rounded)
                has_incoherent, _, incoherent_sentences, _ = self.process_paragraph(
                    self.model, paragraph, 3, cmatrix=matrix_rounded)
                result.append({
                    'status': 'contains_error' if has_missing or has_incoherent else 'ok',
                    'has_missing': has_missing,
                    'missing_sentences': missing_sentences,
                    'has_incoherent': has_incoherent,
                    'incoherent_sentences': incoherent_sentences,
                    'matrix_original':matrix_original.tolist()
                })
            else:
                result.append({
                    'status': 'skipped'
                })

        state['coherence'] = result
        return result

    def create_mask_matrix(self, sizeY, sizeX, distance):
        masked_matrix = [[0 for col in range(sizeX)] for row in range(sizeY)]
        cell_list = []
        for i in range(sizeY):
            for j in range(sizeX):
                if j > i and j-i <= distance:
                    masked_matrix[i][j] = 1
                    cell_list.append([i, j])
                else:
                    masked_matrix[i][j] = 0
        return masked_matrix, cell_list

    def mask_paragraph_matrix(self, matrix, distance):
        masked_matrix = [row[:] for row in matrix]
        height = len(masked_matrix)
        width = len(masked_matrix[0])
        cell_list = []
        for i in range(height):
            for j in range(width):
                if j > i and j-i <= distance:
                    if masked_matrix[i][j] > 0.5:
                        masked_matrix[i][j] = 1
                        cell_list.append([i, j])
                    else:
                        masked_matrix[i][j] = 0
                else:
                    masked_matrix[i][j] = 0
        return masked_matrix, cell_list

    def cluster_paragraph_matrix(self, number_elements, edges):
        elements = [[i] for i in range(number_elements)]

        def find(i):
            for arr in elements:
                if i in arr:
                    return arr
            return None

        def union(set1, set2):
            new_set = [*set1, *set2]
            elements.remove(set1)
            elements.remove(set2)
            elements.append(new_set)

        for edge in edges:
            if find(edge[0]) != find(edge[1]):
                union(find(edge[0]), find(edge[1]))

        return elements

    def create_paragraph_matrix(self, model, paragraph, mask=None):

        if mask is not None:
            height = len(mask)
            width = len(mask[0])
            pair_index = []
            first_sentences = []
            second_sentences = []
            for i in range(height):
                for j in range(width):
                    if mask[i][j] == 1:
                        pair_index.append(i*width+j)
                        first_sentences.append(paragraph[i])
                        second_sentences.append(paragraph[j])
        else:
            raise Exception("Create paragraph matrix without mask deprecated")
            # Orignally used in training
            # combinations = list(product(paragraph, paragraph))
            # first_sentences = [pair[0] for pair in combinations]
            # second_sentences = [pair[1] for pair in combinations]

        total_elements = len(second_sentences)
        labels = [0]*total_elements

        # print(paragraph)
        # print(first_sentences)
        # print(second_sentences)

        encodings = self.tokenizer(first_sentences, second_sentences,
                                   truncation=True, padding=True, max_length=64)
        ds = CustomDataset(
            first_sentences, second_sentences, encodings, labels)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        results = []
        with torch.no_grad():
            # with tqdm(iter(loader), total=len(loader),leave=False) as t:
            for it, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

                probs = F.softmax(outputs, dim=1)
                probs = probs[:, 0]
                results.append(probs)

        if mask is not None:
            results = torch.cat(results).detach().cpu().numpy()
            temp = [0]*(width*height)
            temp2 = [0]*(width*height)
            for i, val in enumerate(results):
                temp[pair_index[i]] = 1 if val > 0.5 else 0
                temp2[pair_index[i]] = val

            results = np.reshape(temp, (len(paragraph), len(paragraph)))
            results_before_rounding = np.reshape(temp2, (len(paragraph), len(paragraph)))
            return results, results_before_rounding
        else:
            pass
            # Originally used in training
            # results = torch.cat(results)
            # results = torch.reshape(
            #     results, (len(paragraph), len(paragraph))).detach().cpu().numpy()

        return results

    def process_paragraph(self, model, paragraph, distance=1, cmatrix=None, generate_full_matrix=True):
        if cmatrix is not None:
            matrix = cmatrix
        else:
            if generate_full_matrix:
                matrix, _ = self.create_paragraph_matrix(model, paragraph)
            else:
                masked_arr, cell_list = self.create_mask_matrix(
                    len(paragraph), len(paragraph), distance)
                matrix, _ = self.create_paragraph_matrix(
                    model, paragraph, mask=masked_arr)

        # process the matrix
        masked_arr, cell_list = self.mask_paragraph_matrix(matrix, distance)
        clusters = self.cluster_paragraph_matrix(len(masked_arr), cell_list)
        large_cluster = max(clusters, key=lambda item: len(item))

        unrelated_sentences = [
            e for arr in clusters for e in arr if len(arr) == 1]
        unrelated_sentences = [
            1 if i in unrelated_sentences else 0 for i in range(len(masked_arr))]

        cluster_marking = [0 for i in range(len(masked_arr))]
        for cls_i, clst in enumerate(clusters):
            for i in clst:
                cluster_marking[i] = cls_i

        missing_sentences = [1 if cluster_marking[i] != cluster_marking[i+1] and (len(clusters[cluster_marking[i]]) > 1 or len(
            clusters[cluster_marking[i+1]]) > 1) else 0 for i in range(len(cluster_marking)-1)]

        has_incoherent_sentence = 1 if 1 in unrelated_sentences else 0
        has_missing = 1 if 1 in missing_sentences else 0

        return has_incoherent_sentence, has_missing, unrelated_sentences, missing_sentences
