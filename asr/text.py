from pythainlp.tokenize import word_tokenize


def tokenize_text(text):
    return word_tokenize(text, engine="newmm")


def clean_repeated_words(text):
    tokens = tokenize_text(text)
    cleaned_tokens = _clean_repeated_tokens(tokens)
    cleaned_text = "".join(cleaned_tokens)
    return cleaned_text



# ------------------------------------------
#  Private helpers
# ------------------------------------------


def _clean_repeated_tokens(tokens):

    tokens = tokens.copy()
    sequence_size = len(tokens) // 2
    while sequence_size > 0:
        cur_idx = 0
        while cur_idx < len(tokens) - sequence_size:        
            next_idx = cur_idx + sequence_size              
            cur_text = "".join(tokens[cur_idx : cur_idx + sequence_size])    
            next_text = "".join(tokens[next_idx : next_idx + sequence_size]) 
            if cur_text == next_text:
                tokens = tokens[: cur_idx + sequence_size] + tokens[next_idx + sequence_size :]
            else:
                cur_idx += 1
        sequence_size -= 1
    return tokens

