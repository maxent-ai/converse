import attr
import pandas as pd
import numpy as np
from .utils import load_sentence_transformer, load_spacy
from nltk.tokenize.texttiling import TextTilingTokenizer
from sklearn.metrics.pairwise import cosine_similarity

model = load_sentence_transformer()
nlp = load_spacy()


@attr.s
class SemanticTextSegmention:

    """
    Segment a call transcript based on topics discussed in the call using
    TextTilling with Sentence Similarity via sentence transformer.

    Paramters
    ---------
    data: pd.Dataframe
        Pass the trascript in the dataframe format

    utterance: str
        pass the column name which represent utterance in transcript dataframe

    """

    data = attr.ib()
    utterance = attr.ib(default='utterance')

    def __attrs_post_init__(self):
        columns = self.data.columns.tolist()

    def get_segments(self, threshold=0.7):
        """
        Paramters
        ---------
        threshold: float
            sentence similarity threshold. (used to merge the sentences into coherant segments)

        Return
        ------
        new_segments: list
            list of segments        
        """
        segments = self._text_tilling()
        merge_index = self._merge_segments(segments, threshold)
        new_segments = []
        for i in merge_index:
            seg = ' '.join([segments[_] for _ in i])
            new_segments.append(seg)
        return new_segments

    def _merge_segments(self, segments, threshold):
        segment_map = [0]
        for index, (text1, text2) in enumerate(zip(segments[:-1], segments[1:])):
            sim = self._get_similarity(text1, text2)
            if sim >= threshold:
                segment_map.append(0)
            else:
                segment_map.append(1)
        return self._index_mapping(segment_map)

    def _index_mapping(self, segment_map):
        index_list = []
        temp = []
        for index, i in enumerate(segment_map):
            if i == 1:
                index_list.append(temp)
                temp = [index]
            else:
                temp.append(index)
        index_list.append(temp)
        return index_list

    def _get_similarity(self, text1, text2):
        sentence_1 = [i.text.strip()
                      for i in nlp(text1).sents if len(i.text.split(' ')) > 1]
        sentence_2 = [i.text.strip()
                      for i in nlp(text2).sents if len(i.text.split(' ')) > 2]
        embeding_1 = model.encode(sentence_1)
        embeding_2 = model.encode(sentence_2)
        embeding_1 = np.mean(embeding_1, axis=0).reshape(1, -1)
        embeding_2 = np.mean(embeding_2, axis=0).reshape(1, -1)
        sim = cosine_similarity(embeding_1, embeding_2)
        return sim

    def _text_tilling(self):
        tt = TextTilingTokenizer(w=15, k=10)
        text = '\n\n\t'.join(self.data[self.utterance].tolist())
        segment = tt.tokenize(text)
        segment = [i.replace("\n\n\t", ' ') for i in segment]
        return segment
