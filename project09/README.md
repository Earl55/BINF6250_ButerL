# Introduction
Forward, Backward, and Forward-Backward project is a continuation from project 8 on Hidden Markov Models. Taking observed state probalities to calculate hidden state probalities and the most optimal route.

# Pseudocode
Put pseudocode in this box:

```
FUNCTION FORWARD(observations, initial_probabilities, transition_probabilities, emission_probabilities):

    hidden_states ← list of keys in initial_probabilities

    forward_table ← empty list

    # --- Initialization step ---
    create empty dictionary for time 0
    append it to forward_table

    FOR each state in hidden_states:
        first_observation ← observations[0]
        forward_table[0][state] ← initial_probabilities[state] 
                                   × emission_probabilities[state][first_observation]

    # --- Recursion step ---
    FOR each time t from 1 to length(observations) - 1:
        current_observation ← observations[t]

        create empty dictionary
        append it to forward_table

        FOR each current_state in hidden_states:

            total_probability ← 0

            FOR each previous_state in hidden_states:
                probability ← forward_table[t-1][previous_state]
                              × transition_probabilities[previous_state][current_state]
                              × emission_probabilities[current_state][current_observation]

                total_probability ← total_probability + probability

            forward_table[t][current_state] ← total_probability

    RETURN forward_table

FUNCTION BACKWARD(observations, initial_probabilities, transition_probabilities, emission_probabilities):

    hidden_states ← list of keys in initial_probabilities

    backward_table ← empty list

    # --- Initialization step ---
    create empty dictionary for final time
    append it to backward_table

    FOR each state in hidden_states:
        backward_table[0][state] ← 1

    # --- Recursion step ---
    # iterate from second-to-last observation backwards to the first
    FOR each time index i going backward through observations (from T-2 down to 0):
        next_observation ← observations[i + 1]

        create empty dictionary
        insert it at the front of backward_table

        FOR each current_state in hidden_states:

            total_probability ← 0

            FOR each next_state in hidden_states:
                probability ← backward_table[1][next_state]
                              × transition_probabilities[next_state][current_state]
                              × emission_probabilities[current_state][next_observation]

                total_probability ← total_probability + probability

            backward_table[0][current_state] ← total_probability

    RETURN backward_table
```

# Successes
This project got easier when it came to understanding the algorithm as a whole, for both Viterbi and Forward, Backward, FB.
From thinking of an example observation with hats and happy/sad. It is all about following along and keeping track with where the probalities come from.

# Struggles
I kept confusing myself initiallay with what to ultply and what probalities to add (for FOrward/Backward)

# Personal Reflections
## Group Leader
After getting an understanding with project 08, completing project 09 was not that bad since it is a similar assignment. Like I said before writing down the process to follow along in addition to the class meetings, it really helps to know the gist of what the algorithms should do and how to apply them.


# Generative AI Appendix
I used AI to help write the pseudo code
