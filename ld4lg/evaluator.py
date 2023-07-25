import spacy
from evaluate import load
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer


class TextGenerationEvaluator:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        mode,
        mauve_model_ids=['gpt2-large', 'all-mpnet-base-v2'],
        perplexity_model_ids=['gpt2-large'],
    ):
        assert mode in {'val', 'test'}, f'Invalid mode: {mode}'
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.mode = mode
        self.mauve_model_ids = mauve_model_ids
        self.perplexity_model_ids = perplexity_model_ids

    def __call__(self, output_texts, class_id=None):
        assert isinstance(output_texts, list) and isinstance(output_texts[0], str)
        num_samples = len(output_texts)

        if class_id is not None:
            assert isinstance(class_id, int)
            if self.mode == 'val':
                train_reference_texts = self.train_dataset.filter(lambda x: x['label'] == class_id)['text']
                val_reference_texts = self.val_dataset.filter(lambda x: x['label'] == class_id)['text']
            else:  # self.mode == 'test'
                test_reference_texts = self.test_dataset.filter(lambda x: x['label'] == class_id)['text']
        else:
            if self.mode == 'val':
                train_reference_texts = self.train_dataset['text']
                val_reference_texts = self.val_dataset['text']
            else:  # self.mode == 'test'
                test_reference_texts = self.test_dataset['text']

        self.train_dataset[:num_samples]
        self.val_dataset[:num_samples]
        self.test_dataset[:num_samples]

        metrics = {}

        metrics['perplexity'] = {}
        for perplexity_model_id in self.perplexity_model_ids:
            metrics['perplexity'][perplexity_model_id] = self.compute_perplexity(output_texts, perplexity_model_id)

        metrics['word_count'] = self.compute_word_count(output_texts)

        metrics['diversity'] = self.compute_diversity(output_texts)

        metrics['memorization'] = self.compute_memorization(output_texts, self.train_dataset['text'])

        metrics['mauve'] = {}
        for mauve_model_id in self.mauve_model_ids:
            metrics['mauve']['mauve_model_id'] = {}
            if self.mode == 'val':
                metrics['mauve'][mauve_model_id]['train'] = self.compute_mauve(
                    output_texts,
                    train_reference_texts,
                    mauve_model_id,
                )
                metrics['mauve'][mauve_model_id]['val'] = self.compute_mauve(
                    output_texts,
                    val_reference_texts,
                    mauve_model_id,
                )
            else:  # self.mode == 'test'
                metrics['mauve'][mauve_model_id]['test'] = self.compute_mauve(
                    output_texts,
                    test_reference_texts,
                    mauve_model_id,
                )

        return metrics

    @staticmethod
    def compute_perplexity(texts, model_id):
        perplexity = load('perplexity', module_type='metric')
        results = perplexity.compute(predictions=texts, model_id=model_id)
        return results['mean_perplexity']

    @staticmethod
    def compute_word_count(texts):
        word_count = load('word_count')
        results = word_count.compute(data=texts)
        return results['unique_words']

    @staticmethod
    def compute_diversity(texts, n_list=[2, 3, 4]):
        tokenizer = spacy.load('en_core_web_sm').tokenizer
        tokens_list = []
        for text in texts:
            tokens_list.append([str(token) for token in tokenizer(text)])

        ngram_dict = {
            n: {
                'unique': set(),
                'count': 0,
            }
            for n in n_list
        }
        for n in n_list:
            for tokens in tokens_list:
                ngram = ngrams(tokens, n)
                ngram_dict[n]['unique'].update(ngram)
                ngram_dict[n]['count'] += len(list(ngram))

        diversity = 1
        for n in n_list:
            unique_ngrams = len(ngram_dict[n]['unique']) / ngram_dict['count'][n]
            diversity *= unique_ngrams

        return diversity

    @staticmethod
    def compute_memorization(texts, train_texts, n=4):
        tokenizer = spacy.load('en_core_web_sm').tokenizer

        train_unique_ngrams = set()
        for text in train_texts:
            tokens = [str(token) for token in tokenizer(text)]
            ngram = ngrams(tokens, n)
            train_unique_ngrams.update(ngram)

        total_ngrams = 0
        duplicate_ngrams = 0
        for text in texts:
            tokens = [str(token) for token in tokenizer(text)]
            ngram_list = list(ngrams(tokens, n))
            total_ngrams += len(ngram_list)
            for ngram in ngram_list:
                if ngram in train_unique_ngrams:
                    duplicate_ngrams += 1
        memorization = duplicate_ngrams / total_ngrams

        return memorization

    @staticmethod
    def compute_mauve(texts, reference_texts, model_id):
        assert len(texts) == len(reference_texts)
        mauve = load('mauve')

        if model_id == 'all-mpnet-base-v2':
            model = SentenceTransformer(model_id).cuda()
            # Sentences are encoded by calling model.encode()
            text_features = model.encode(texts)
            reference_text_features = model.encode(reference_texts)
            results = mauve.compute(
                predictions=texts,
                p_features=text_features,
                references=reference_texts,
                q_features=reference_text_features,
                featurize_model_name=model_id,
                max_text_length=256,
                mauve_scaling_factor=8,
            )
        elif model_id == 'gpt2-large':
            results = mauve.compute(
                predictions=texts,
                references=reference_texts,
                featurize_model_name=model_id,
                max_text_length=256,
            )
        else:
            raise NotImplementedError(f'Model {model_id} not supported')

        return results['mauve']
