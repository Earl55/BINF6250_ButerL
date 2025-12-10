from HMM import HMM

from copy import deepcopy


standard_alphabets = list("ACDEFGHIKLMNPQRSTVWY")


class ProfileHMM:

    def __init__(self, sequences, gap_char="-", alphabet=standard_alphabets):

        self.seqs = sequences
        self.seq_len = len(self.seqs[0])
        self.col_len = len(self.seqs)
        self.gap_char = gap_char
        self.alphabet = alphabet

        self.hmm = None

        for seq in self.seqs:
            if len(seq) != self.seq_len:
                raise ValueError("All sequences must have the same length.")
        
        self.__col_is_match = self.__classify_col_is_match()
        self.profile_length = sum(self.__col_is_match)

        self.__estimate_params()
    
    def __classify_col_is_match(self):

        col_classes = []

        for col_idx in range(self.seq_len):

            col = [seq[col_idx] for seq in self.seqs]
            gap_count = col.count(self.gap_char)

            col_classes.append(gap_count < (self.col_len/2))
        
        return col_classes
    
    def __build_hidden_states_and_prob_dicts(self):

        # hidden states
        self.hiddenstates = ["M0", "I0"]
        for i in range(1, self.profile_length+1):
            self.hiddenstates += [f"M{i}", f"I{i}", f"D{i}"]
        self.hiddenstates += [f"M{self.profile_length+1}"]
    
        # trans probs

        self.trans_probs = {}
        for state in self.hiddenstates:
            self.trans_probs[state] = {}
            for state2 in self.hiddenstates:
                self.trans_probs[state][state2] = 0

        # self.trans_probs = {}
        # for i in range(self.profile_length + 2):

            # m_state = f"M{i}"
            # if m_state not in self.trans_probs:
            #     self.trans_probs[m_state] = {}

            # added_states = [f"M{i+1}", f"I{i}", f"D{i+1}"]
            # for added_state in added_states:
            #     if added_state in self.hiddenstates:
            #         self.trans_probs[m_state][added_state] = 0
            
            # i_state = f"I{i}"
            # if i_state not in self.trans_probs:
            #     self.trans_probs[i_state] = {}

            # added_states = [f"I{i}", f"M{i+1}", f"D{i+1}"]
            # for added_state in added_states:
            #     if added_state in self.hiddenstates:
            #         self.trans_probs[i_state][added_state] = 0
            
            # d_state = f"D{i}"
            # if d_state not in self.trans_probs:
            #     self.trans_probs[d_state] = {}

            # added_states = [f"D{i+1}", f"M{i+1}", f"I{i}"]
            # for added_state in added_states:
            #     if added_state in self.hiddenstates:
            #         self.trans_probs[d_state][added_state] = 0
        
        # emit probs
        self.emit_probs = {}
        for state in self.hiddenstates:
            if state not in self.emit_probs:
                self.emit_probs[state] = {}
            for emission in self.alphabet:
                self.emit_probs[state][emission] = 0

        # init probs
        self.init_probs = {state: 0 for state in self.hiddenstates}

    
    def __label_sequence_path(self, seq):

        path_states = []
        path_emissions = []

        k = 0

        for col, match in zip(range(self.seq_len), self.__col_is_match):

            if match:

                k += 1

                if seq[col] == self.gap_char:

                    path_states.append(f'D{k}')
                    path_emissions.append(None)
                else:

                    path_states.append(f'M{k}')
                    path_emissions.append(seq[col])

            else:

                if seq[col] == self.gap_char:
                    continue
                else:

                    path_states.append(f'I{k}')
                    path_emissions.append(seq[col])

        path_states.append(f'M{k+1}')
        path_emissions.append(None)
        return path_states, path_emissions

    def __estimate_params(self):

        self.__build_hidden_states_and_prob_dicts()

        trans_counts = deepcopy(self.trans_probs)
        emit_counts = deepcopy(self.emit_probs)
        init_counts = deepcopy(self.init_probs)

        global_residue_counts = {}
        insert_residue_counts = {}

        for seq in self.seqs:

            seq_states, seq_emissions = self.__label_sequence_path(seq)

            for i, (state, emission) in enumerate(zip(seq_states, seq_emissions)):

                if emission is None:
                    continue

                if emission not in self.alphabet:
                    continue

                emit_counts[state][emission] += 1
                global_residue_counts[emission] = global_residue_counts.get(emission, 0) + 1

                if state.startswith("I"):
                    insert_residue_counts[emission] = insert_residue_counts.get(emission, 0) + 1
                
                if i < len(seq_states)-1:
                    next_state = seq_states[i+1]
                    trans_counts[state][next_state] += 1.0
                
        bg_counts = insert_residue_counts
        
        alpha = 1
        emit_probs = {}

        for state in self.hiddenstates:

            emit_probs[state] = {}

            if state.startswith("M"):

                total_state_count = sum(emit_counts[state].values())

                for a in self.alphabet:
                    count = emit_counts[state].get(a, 0)
                    emit_probs[state][a] = (count + alpha) / (total_state_count + (alpha*len(self.alphabet)))

            elif state.startswith("I"):

                total_state_count = sum(bg_counts.values())

                for a in self.alphabet:
                    count = bg_counts.get(a, 0)
                    emit_probs[state][a] = (count) / (total_state_count)
            
            else:

                emit_probs[state] = {a: 1/len(self.alphabet) for a in self.alphabet}
                continue
    
        trans_probs = {}

        for state in self.hiddenstates:

            total = sum(trans_counts[state].values()) + (len(trans_counts[state]) * alpha)
            trans_probs[state] = {next_state: (trans_counts[state][next_state]+alpha)/total for next_state in trans_counts[state]}
    
        init_probs = init_counts
        init_probs["M0"] = 1

        self.emit_probs = emit_probs
        self.trans_probs = trans_probs
        self.init_probs = init_probs

        self.hmm = HMM(alphabet=self.alphabet, hidden_states=self.hiddenstates, init_probs=self.init_probs,
                       trans_probs=self.trans_probs, emit_probs=self.emit_probs)
    
    def baulm_welch_retrain(self, sequences=None):

        if self.hmm is None:
            raise ValueError("HMM model not initialized")

        if sequences is None:
            sequences = self.seqs

        self.hmm.baum_welch(sequences=sequences)
    
    def total_forward_probability(self, sequence):

        prob, _ = self.hmm.forward(sequence)
        return prob
    
    def most_likely_path(self, sequence):

        path = self.hmm.viterbi(sequence)
        return path
