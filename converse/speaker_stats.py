import attr
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from .utils import load_spacy

nlp = load_spacy()

@attr.s
class SpeakerStats:
    """
    For given transcript, calculate the various lingustic attributes for different speakers.

    Paramters
    ---------
    data: pd.Dataframe
        Pass the trascript in the dataframe format
    
    speaker: str
        pass the column name which represent speaker name/id in transcript dataframe

    utterance: str
        pass the column name which represent utterance in transcript dataframe

    

    """
    data = attr.ib()
    speaker = attr.ib(default='speaker')
    utterance = attr.ib(default='utterance')
    
    def __attrs_post_init__(self):
        columns = self.data.columns.tolist()
        
        if self.speaker not in columns:
            raise ValueError("Please pass proper speaker column")
        
        if self.utterance not in columns:
            raise ValueError("Please pass proper speaker column")
            
    
    def get_stats(self, n_topic=2):
        """
        parameters
        ----------
        n_topic: int
            Define the maximum number of speaker stat you want to identify.
        
        Returns
        -------
        return_dict: dict
            Dictionary of user stats.
        """

        self.data['speaker_stats'] = self.data[self.utterance].apply(self.get_text_summary)
        return_dict = {}
        for spk, df in self.data.groupby(by=self.speaker):
            df = df[~df.speaker_stats.isin([None,"Informal, personal"])]
            _ = Counter(df.speaker_stats.tolist()).most_common(n_topic)
            _ = [i for (i,j) in _]
            return_dict[spk] = _
        return return_dict
    
    def _tag_stats(self, tagged_input):
        num_pronouns = 0
        num_prp = 0
        num_articles = 0
        num_past = 0
        num_future = 0
        num_prep = 0

        for token in tagged_input:
            tag = token.tag_
            if tag in ["PRP", "PRP$", "WP", "WP$"]:
                num_pronouns += 1

            if tag in ["PRP"]:
                num_prp += 1

            if tag in ['DT']:
                num_articles += 1            

            if tag in ['VBD', 'VBN']:
                num_past += 1

            if tag in ['MD']:
                num_future +=1

            if tag in ['IN']:
                num_prep+=1

        return_dict = dict(num_pronouns=num_pronouns, num_prp=num_prp, num_articles=num_articles,
                          num_past=num_past, num_future=num_future, num_prep=num_prep)
        return return_dict

    def _word_counter(self, text):
        count = 0
        for i in text:
            if not i.is_punct:
                count += 1
        return count

    def _avg_words_per_sentence(self, text):
        count_list = []
        for i in text.sents:
            c = self._word_counter(i)
            count_list.append(c)
        return round(np.mean(np.array(count_list)),2)

    def _count_negation(self, text):
        temp = [tok for tok in text if tok.dep_ == 'neg']
        return len(temp)

    def get_lingustic_stats(self, text):
        text = nlp(text)
        stats = self._tag_stats(text)
        stats['num_words'] = self._word_counter(text)
        stats['wps'] = self._avg_words_per_sentence(text)
        stats['num_negations'] = self._count_negation(text)
        return stats   
    
    def get_text_summary(self, text):
        t = self.get_lingustic_stats(text)
        temp = self._get_correlation(t)
        if len(temp):
            temp = temp[0]
        else:
            temp = None
        return temp
    
    def _get_correlation(self, data, num_words_threshold=100, wps_threshold=20):
        informative_correlates = []
        psychological_correlates = {}
        psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
        psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
        psychological_correlates["num_pronouns"] = "Informal, personal"
        psychological_correlates["num_prp"] = "Personal, social"
        psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
        psychological_correlates["num_past"] = "Focused on the past"
        psychological_correlates["num_future"] = "Future and goal-oriented"
        psychological_correlates["num_prep"] = "Education, concern with precision"
        psychological_correlates["num_negations"] = "Inhibition"
        # Set thresholds
        if data['num_words'] > num_words_threshold:
            informative_correlates.append(psychological_correlates['num_words'])

        if data['wps'] > wps_threshold:
            informative_correlates.append(psychological_correlates['wps'])

        d = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

        all_visited = False
        while(len(informative_correlates) < 3):
            for index,(i,j) in enumerate(d.items()):
                if j ==0:
                    continue
                if i == 'num_prep' and d['num_pronouns'] == d['num_prep'] and len(informative_correlates)==2:
                    informative_correlates.append(psychological_correlates['num_pronouns'])
                elif i not in ['num_words', 'wps']:
                    informative_correlates.append(psychological_correlates[i])
            if index+1 == len(d):
                break

        return informative_correlates
