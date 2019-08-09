import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm


class emrQA(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'emrQA'

    @overrides
    def build_header(self, preprocessor, contexts, split, dataset_version, dataset_flavor, dataset_specific_props):

        header = {
            "dataset_name": self.DATASET_NAME,
            "version": dataset_version,
            "flavor": dataset_flavor,
            "split": split,
            "dataset_url": "https://rajpurkar.github.io/SQuAD-explorer/",
            "license": "http://creativecommons.org/licenses/by-sa/4.0/legalcode",
            "data_source": "Wikipedia",
            "context_answer_detection_source": self.DATASET_NAME,
            "tokenization_source": "MultiQA",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_qas_with_gold_answers": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self, preprocessor, split, sample_size, dataset_version, dataset_flavor, dataset_specific_props, input_file):
        single_file_path = "emrqa.json"

        with open(single_file_path, 'r') as myfile:
            original_dataset = json.load(myfile)

        contexts = []
        data = original_dataset['data']

        counter = 0
        for topic in tqdm.tqdm(data, total=len(data), ncols=80):
            for paragraph in topic['paragraphs']:
                qas = []
                for qa in paragraph['qas']:
                    counter +=1
                    new_qa = {'qid':self.DATASET_NAME + '_q_' + str(counter),
                                'question':qa['question'][0]}
                    answer_candidates = []
                    
                    for answer_candidate in qa['answers']:
                        answer_candidates.append({'extractive':{"single_answer":{"answer": answer_candidate['text'],
                            "instances": [{'doc_id':0,
                                        'part':'text',
                                        'start_byte':answer_candidate['answer_start'],
                                        'text':answer_candidate['text']}]}}})
                    new_qa['answers'] = {"open-ended": {'answer_candidates': answer_candidates}}
                    qas.append(new_qa)
                paragraph['context'] = "".join(paragraph['context'])
                paragraph['context'] = paragraph['context'].replace("\n"," ")
                contexts.append({"id": self.DATASET_NAME + '_'  + gen_uuid(),
                                "context": {"documents": [{"text": paragraph['context'],
                                                            "title":topic['title']}]}, "qas": qas})
        if sample_size != None:
            contexts = contexts[0:sample_size]

        contexts = preprocessor.tokenize_and_detect_answers(contexts)
        
        return contexts
