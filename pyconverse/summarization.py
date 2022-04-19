import attr
from .utils import load_summarization_model
from .segmentation import SemanticTextSegmentation
from tqdm import tqdm


@attr.s
class TranscriptSummarization:

    """
    Apply the Abstractive summerization on Transcript.

    Paramters
    ---------
    data: pd.Dataframe
        Pass the trascript in the dataframe format

    utterance: str
        pass the column name which represent utterance in transcript dataframe

    speaker: str
        pass the column name which represent speaker name/id in transcript dataframe

    similarity_threshold: float
        pass the float value between 0 to 1

    """

    data = attr.ib()
    utterance = attr.ib(default='utterance')
    speaker = attr.ib(default='speaker')
    similarity_threshold = attr.ib(default=0.65)
    _summary_model = attr.ib(default=None)
    _segments = attr.ib(default=[])

    def __attrs_post_init__(self):
        self.data[self.speaker] = self.data[self.speaker].astype(str)
        self._summary_model = load_summarization_model()

    def _create_segments(self):
        tt = SemanticTextSegmentation(self.data, self.utterance)
        self._segments = tt.get_segments(self.similarity_threshold)

        segment_index = 0
        segment_idx = []
        for index, row in self.data.iterrows():
            if row[self.utterance] in self._segments[segment_index]:
                segment_idx.append(segment_index)
            elif row[self.utterance] in self._segments[segment_index+1]:
                segment_index += 1
                segment_idx.append(segment_index)
        self.data["segment_idx"] = segment_idx
        return

    def _get_segment_summary(self, segments):
        conv = segments[self.speaker] + " : "+segments[self.utterance]
        text = "\n".join(conv.tolist())

        if len(segments) < 5 and len(text.split(' ')) < 150:
            return ""

        try:
            summary = self._summary_model(text)
        except:
            max_len = int(len(text)/3)
            summary = self._summary_model(text[:max_len])

        return summary[0]["summary_text"]

    def get_summary(self):
        """
        Return the transcript summary.

        """
        self._create_segments()
        summary = ""
        for i, sgmt in tqdm(self.data.groupby("segment_idx")):
            try:
                summary += self._get_segment_summary(sgmt)
            except:
                continue

        return summary
