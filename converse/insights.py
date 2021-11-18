import attr
import spacy
import pandas as pd
from ._const import backchannel
from .utils import load_sentence_transformer, remove_punct, load_spacy, load_zeroshot_model


nlp = load_spacy()
classifier = load_zeroshot_model()


@attr.s
class Callyzer:

    """
    For given transcript, calculate the various lingustic and numerical attributes at speaker level.

    Paramters
    ---------
    data: pd.Dataframe
        Pass the trascript in the dataframe format

    speaker: str
        pass the column name which represent speaker name/id in transcript dataframe

    utterance: str
        pass the column name which represent utterance in transcript dataframe

    startime: str
        pass the column name which represent start time for utterance in transcript dataframe

    endtime: str
         pass the column name which represent end time for utterance in transcript dataframe
    """

    data = attr.ib()
    utterance = attr.ib(default='utterance')
    speaker = attr.ib(default='speaker')
    startime = attr.ib(default='startTime')
    endtime = attr.ib(default='endTime')

    def __attrs_post_init__(self):
        columns = self.data.columns.tolist()
        if self.speaker not in columns:
            raise ValueError("Please pass proper speaker column")

        if self.speaker not in columns:
            raise ValueError("Please pass proper utterance column")

        if self.startime not in columns:
            raise ValueError(
                "Please pass proper starttime column. We need to calculate feature like silence, interruption etc.")

        if self.endtime not in columns:
            raise ValueError(
                "Please pass proper endtime column. We need to calculate feature like silence, interruption etc")

    def get_interruption(self, threshold=1):
        """
        Parameters
        ----------
        threshold: int
            Minimun overlap seconds between two users to consider as interrptions.

        Returns
        -------
        return_dict: dict
            Returns the speaker wise interruption with time stamps..

        """
        return_dict = {}
        start_times = self.data[self.starttime].tolist()
        end_times = self.data[self.endtime].tolist()
        interrupt_ids = []

        for index, (et, st) in enumerate(zip(end_times[:-1], start_times[1:])):
            if et - st >= threshold:
                interrupt_ids.append(index)

        return_dict = self._get_channel_details(interrupt_ids)
        return_dict['total_interruption'] = len(interrupt_ids)
        return return_dict

    def get_silence(self, threshold=1):
        """
        Parameters
        ----------
        threshold: int
            Minimun non-talk seconds between two users to consider as silence.

        Returns
        -------
        return_dict: dict
            Returns the speaker wise silence with time stamps.

        """
        return_dict = {}
        start_times = self.data[self.startime].tolist()
        end_times = self.data[self.endtime].tolist()
        interrupt_ids = []

        for index, (et, st) in enumerate(zip(end_times[:-1], start_times[1:])):
            if st - et >= threshold:
                interrupt_ids.append(index)
        return_dict = self._get_channel_details(interrupt_ids)
        return_dict['total_interruption'] = len(interrupt_ids)
        return return_dict

    def convert_at_turn(self):
        """
        Convert utterance to turns
        """
        df = self._collapse_df()
        return df

    def _collapse_df(self,):
        tp = []
        df = self.get_turn_ids()
        for i, j in self.data.groupby(by="turn_id"):
            temp = j.iloc[0]
            if len(j) > 1:
                temp[self.utterance] = ' '.join(j[self.utterance].tolist())
                temp[self.endtime] = j.iloc[-1][self.endtime]
            tp.append(temp)
        tp = pd.concat(tp, axis=1).T
        return tp

    def get_turn_ids(self, in_place=True):
        turn = 0
        channel = None
        return_data = None
        turn_list = []

        for i, j in self.data.iterrows():
            if channel is None:
                channel = j[self.speaker]

            if channel != j[self.speaker]:
                channel = j[self.speaker]
                turn += 1
            turn_list.append(turn)

        if in_place:
            self.data['turn_id'] = turn_list
            return self.data
        else:
            return turn_list

    def _get_channel_details(self, id_list):
        channels = self.data[self.speaker].unique()
        return_dict = {k: dict(metadata=[], count=0) for k in channels}
        for idx in id_list:
            ch = df.iloc[idx+1]
            channel = ch.get(self.speaker)
            _ = dict(start_time=ch.get(self.startime),
                     end_time=ch.get(self.endtime), index=idx+1)

            update_data = return_dict[channel]
            update_data['count'] = update_data['count'] + 1
            update_data['metadata'] = update_data['metadata']+[_]
            return_dict[channel] = update_data
        return return_dict

    def tag_questions(self, inplace=True):
        """
        For utterance, tag if it a question or not.

        Parameters
        ----------
        inplace: bool
           Add the new column in to dataframe if inplace is True.

        Returns
        -------
        questions: pd.dataframe or pd.Series
            Returns the dataframe or series 
        """
        questions = self.data[self.utterance].apply(self._is_text_question)
        if inplace:
            self.data['is_question'] = questions
            return self.data
        else:
            return questions

    def tag_backchannel(self, type='default', inplace=True, model_name='all-MiniLM-L6-v2'):
        """
        For utterance, tag if it a backchannel or not.

        Parameters
        ----------
        type: str, default or nlp
            type of back channel detection.

        inplace: bool
           Add the new column in to dataframe if inplace is True.


        model_name: str 
            Pass the any sentence transfromer model name which use similary when backchannel type is nlp

        Returns
        -------
        questions: pd.dataframe or pd.Series
            Returns the dataframe or series 
        """
        if type == 'default':
            backchannel = self.data[self.utterance].apply(
                lambda x: self._is_backchannel(x, backchannel))

        elif type == "nlp":
            backchannel = self._nlp_backchannel(
                self.data[self.utterance].tolist(), model_name)

        else:
            raise ValueError(
                "Please pass backchannel either as `default` or `nlp`")

        if inplace:
            self.data['is_backchannel'] = backchannel
            return self.data
        else:
            return backchannel

    def _is_backchannel(self, text, constant_list):
        is_back_ch = False
        clean_text = remove_punct(text)

        if text in constant_list or text.lower() in constant_list:
            is_back_ch = True

        if clean_text in constant_list or clean_text.lower() in constant_list:
            is_back_ch = True

        return is_back_ch

    def _nlp_backchannel(self, utterances, model='all-MiniLM-L6-v2'):
        if isinstance(utterances, str):
            utterances = [utterances]

        return_list = [False]*len(utterances)
        model = load_sentence_transformer(model)
        back_channel = ["hmmm", "yeah okay",
                        "oh really", "oh wow", "thats good", "cool"]
        back_ch_vect = model.encode(back_channel)
        back_ch_vect = [np.mean(back_ch_vect, axis=0)]
        utterance_vect = model.encode(utterances)
        sim = cosine_similarity(back_ch_vect, utterance_vect)[0]
        for idx, i in enumerate(sim):
            if i >= 0.60 or abs(i-0.60) < 0.05:
                return_list[idx] = True
        return return_list

    def _is_text_question(self, text):
        if len(text) < 10:
            return False
        doc = nlp(text)
        wh_tags = ["WDT", "WP", "WP$", "WRB"]
        wh_words = [t for t in doc if t.tag_ in wh_tags]
        start_with_wh = wh_words and wh_words[0].i == 0
        has_wh = wh_words and wh_words[0].head.dep_ == "prep"
        pseudo_wh = wh_words and wh_words[0].head.dep_ in ["csubj", "advcl"]
        if pseudo_wh:
            return False
        elif start_with_wh or pseudo_wh:
            return True
        else:
            return False

    def tag_emotion(self, inplace=True):
        """
        For utterance, tag what emotion we found. 
        Emotions that we identify here: 
            ['Surprised', 'Angry', 'Sad', 'Annoyed', 'Lonely', 
             'Guilty', 'Impressed', 'Disgusted', 'Confident', 'Anxious',
             'Joyful', 'Jealous','Caring', 'Sentimental','Neutral']

        Parameters
        ----------
        inplace: bool
           Add the new column in to dataframe if inplace is True.

        Returns
        -------
        questions: pd.dataframe or pd.Series
            Returns the dataframe or series 
        """

        candidate_labels = ['Surprised', 'Angry', 'Sad', 'Annoyed', 'Lonely',
                            'Guilty', 'Impressed', 'Disgusted', 'Confident', 'Anxious',
                            'Joyful', 'Jealous', 'Caring', 'Sentimental', 'Neutral']

        texts = self.data[self.utterance].tolist()
        classes = self._classifier(texts, candidate_labels)
        if inplace:
            self.data['emotion'] = classes
            return self.data
        else:
            return classes

    def tag_empathy(self, inplace=True):
        """
        Tag if the utterance is empathetic or not,
        """
        candidate_labels = ['empathy', 'non_empathetic', 'neutral']
        texts = self.data[self.utterance].tolist()
        classes = self._classifier(texts, candidate_labels)
        if inplace:
            self.data['is_empathy'] = classes
            return self.data
        else:
            return classes

    def _classifier(self, texts, candidate_labels):
        if isinstance(texts, str):
            texts = [texts]

        return_data = []
        for text in tqdm(texts):
            resp = classifier(text, candidate_labels=candidate_labels)
            if resp['scores'][0] >= 0.45:
                return_data.append(resp['labels'][0])
            else:
                return_data.append("not found")
        return return_data
