import os
import pickle
import random
import spacy
import torch

from collections import Counter, defaultdict
from tqdm import tqdm

from src.secret import ROOT, RAW_DATASTORE_DIR  # local paths, replace with your own

methods = ["original", "adjusted"]  # choose between "original" or "adjusted"
METHOD = methods[1]
SAVE = False # whether to save the num_data or not
DATASET = "commonvoice" # ensures there are no mistakes down the line

# get tokenizer and EOT token. Needed to match tokens to words later
with open(os.path.join(ROOT, "whisper_tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
eot = tokenizer.special_tokens["<|endoftext|>"]

# get raw data
with open(os.path.join(RAW_DATASTORE_DIR, f"datastore_raw_data_{DATASET}_partNone_large-v3_lcFalse.pkl"), "rb") as f:
    data = pickle.load(f)
    keys = data['hidden_states']
    vals = data['idxs'] # this is the same as the token ids
    flat_token_ids = vals.flatten().tolist()

# initialise spacy tagger
nlp = spacy.load("nl_core_news_sm")
nlp.max_length = 3_000_000

# *-----------------*
# | Original method |
# *-----------------*
if METHOD == "original":
    # Laad originele zinnen
    with open(os.path.join(ROOT, DATASET, "train_ref_texts.pkl"), "rb") as f:
        train_texts = pickle.load(f)
    
    # POS-tag alle zinnen
    docs = list(nlp.pipe(train_texts, disable=["ner", "parser"], batch_size=1000))

    # Bepaal per token of het een 'NUM' is
    num_token_infos = []  # (zin_index, token_text, token_index_in_zin)

    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            if token.pos_ == "NUM":
                num_token_infos.append((i, token.text, j))

    print("Aantal NUM-tokens gevonden:", len(num_token_infos))
    print("Voorbeelden:", num_token_infos[:5])

    for i in random.sample(num_token_infos, 5):
        zin_index, token_text, token_pos_in_zin = i
        print(f"\n Zin #{zin_index}")
        print("Zin:", train_texts[zin_index])
        print("Gevonden NUM-token:", token_text)
        print("Volledige POS-tagging:")
        print([f"{tok.text} ({tok.pos_})" for tok in docs[zin_index]])

    num_token_words = [t[1] for t in num_token_infos]
    counter = Counter(num_token_words)

    print("\nTop 20 meest voorkomende NUM-tokens:")
    for token, count in counter.most_common(20):
        print(f"{token}: {count}")

    with open(os.path.join(ROOT, DATASET, "train_ref_tokens_texts.pkl"), "rb") as f:
        token_texts = pickle.load(f)

    # Controleer structuur
    print("Aantal zinnen:", len(token_texts))
    print("Type van eerste item:", type(token_texts[0]))

    num_indices = []  # globale indices voor NUM-posities
    global_index = 0

    for i, tokens in enumerate(token_texts):  # per zin
        num_positions = [info[2] for info in num_token_infos if info[0] == i]

        # check per token of het een NUM-token is
        for local_idx, token in enumerate(tokens):
            if local_idx in num_positions:
                num_indices.append(global_index + local_idx)
        global_index += len(tokens)

    print("Aantal gevonden NUM-indices:", len(num_indices))

    print("Shape van keys:", keys.shape)
    print("Aantal vals:", len(vals))

    num_keys = keys[num_indices]  # torch tensor
    num_vals = [vals[i] for i in num_indices]
    num_token_ids = [flat_token_ids[i] for i in num_indices]

    print("Tokens extracted:")
    print([tokenizer.decode(tok) for tok in num_vals[:20]])

# *-----------------*
# | Adjusted method |
# *-----------------*
else:
    # Load reference texts that for spacy to parse
    with open(os.path.join(ROOT, DATASET, "train_ref_texts.pkl"), "rb") as f:
        train_texts = pickle.load(f)
        train_texts = [ref.strip() for ref in train_texts]

    # POS-tag per sentence
    pos_tags = []        # will be list of POS tags for each token in *flat_token_ids*
    for i, ref in tqdm(enumerate(train_texts), total=len(train_texts)):
        # Encoding could also be done by looping over flat_token_ids until you find an EOT, but this is cleaner
        tokens = tokenizer.encode(" "+ref)+[eot]

        doc = nlp(str(ref.strip()))

        # This loop is a bit hard to parse, but here's the gist:
        # - We loop over the *tokens* in the *encoded sentence*
        # - Current word in spacy doc is being tracked by:
        #   - j: index in the spacy doc
        #   - nlp_txt: text of the current word in the spacy doc
        #   - nlp_tag: POS tag of the current word in the spacy doc
        # - For every token, that token's text is removed from the current nlp_txt (by length)
        #   - Example: tok = "de", nlp_txt = "degelijk" -> nlp_txt becomes "gelijk"
        # - Once nlp_txt is empty, we move to the next word in the spacy doc by:
        #   - Incrementing j
        #   - Setting nlp_txt to the text of the next word in the spacy doc
        #   - Setting nlp_tag to the POS tag of the next word in the spacy doc
        # - This ensures that each word's POS tag is assigned to all tokens that make up that word
        # - The loop also handles special cases where the token text does not match the nlp_txt exactly, such as punctuation marks or special characters
        #   - These tokens suck: "±", "’", "”", "´", "“", "Ţ", "‘"
        #   - They're made up of two tokens, but decoding each individually doesn't correspond directly to text
        #   - And so "removing" them by length from nlp_txt doesn't make sense
        cur_pos_tags = [] # pos tags for current sentence

        # keeping track of the current word and POS-tag in the spacy doc
        j = 0
        nlp_txt = doc[j].text
        nlp_tag = doc[j].pos_

        # loop over tokens
        for i_tok, tok in enumerate(tokens[:-1]):
            dec_tok = tokenizer.decode([tok]) # current token as text

            # if we've reached the end of the current word in the spacy doc, go to the next word
            # (checking that the current token text is not in the nlp_txt
            # and that it is not one of the *special cases*)
            if dec_tok.strip() not in nlp_txt \
            and not (nlp_txt == "±" \
            and (
                "±" in tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or "±" in tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )) \
            and not ("’" in nlp_txt \
            and (
                "’" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or "’" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1]) \
                or " ’" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or " ’" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )) \
            and not ("”" in nlp_txt \
            and (
                "”" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or "”" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1]) \
                or " ”" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or " ”" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )) \
            and not ("´" in nlp_txt \
            and (
                " ´" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or " ´" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )) \
            and not ("“" in nlp_txt \
            and (
                "“" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or "“" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1]) \
                or " “" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or " “" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )) \
            and not ("Ţ" in nlp_txt \
            and (
                "Ţ" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or "Ţ" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1]) \
                or " Ţ" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or " Ţ" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )) \
            and not ("‘" in nlp_txt \
            and (
                "‘" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or "‘" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1]) \
                or " ‘" == tokenizer.decode(tokens[i_tok:i_tok+2]) \
                or " ‘" == tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])
            )):
                # print("dec tok:", dec_tok.strip(), "---", "nlp_txt:", nlp_txt, "---", tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1]), "---", tokenizer.decode(tokens[i_tok:i_tok+2])) # debugging
                j += 1
                nlp_txt = doc[j].text
                nlp_tag = doc[j].pos_
                # print(f"new nlp_txt: {nlp_txt}") # debugging
            # print(f"{dec_tok:<20} {nlp_txt:<20} {nlp_tag:<10} {j:>3}") # debugging
            # print(nlp_txt, dec_tok) # debugging

            # add POS tag of current token
            cur_pos_tags.append(nlp_tag)

            # remove part of nlp_txt that corresponds to the current token, handling special cases
            if ("±" not in nlp_txt) \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != "’") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != " ’") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != "”") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != " ”") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != " ´") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != "“") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != " “") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != "Ţ") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != " Ţ") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != "‘") \
            and (tokenizer.decode(tokens[i_tok:i_tok+2]) != " ‘"):
                # print(f"check: {tokenizer.decode(tokens[max(0, i_tok-1):i_tok+1])}, {tokenizer.decode(tokens[i_tok:i_tok+2])}, {nlp_txt}") # debugging
                if ("’" in tokens[max(0, i_tok-1):i_tok+1]) \
                or ("”" in tokens[max(0, i_tok-1):i_tok+1]) \
                or (" ´" in tokens[max(0, i_tok-1):i_tok+1]) \
                or ("“" in tokens[max(0, i_tok-1):i_tok+1]) \
                or ("Ţ" in tokens[max(0, i_tok-1):i_tok+1]) \
                or ("‘" in tokens[max(0, i_tok-1):i_tok+1]):
                    len_to_strip = 1
                else:
                    len_to_strip = len(dec_tok.strip())
                nlp_txt = nlp_txt[len_to_strip:]
                # print(f"check 2: {nlp_txt}") # debugging
                # print()

        # add EOT tag at end of sentence
        cur_pos_tags.append("EOT")

        # *check* that the number of POS tags matches the number of tokens, and add cur_pos_tags to pos_tags
        assert len(cur_pos_tags) == len(tokens), f"{len(cur_pos_tags)} != {len(tokens)}, {i}"
        pos_tags += cur_pos_tags

    # check that we ended up with the same number of POS tags as tokens
    assert len(pos_tags) == len(flat_token_ids), f"{len(pos_tags)} != {len(flat_token_ids)}"

    # find all NUM tokens in the flat_token_ids
    num_token_infos = []  # (token_index, token_text, POS-tag)
    tok_freqs = defaultdict(int) # keep track of num token frequencies
    for i, (tok, pos) in tqdm(enumerate(zip(flat_token_ids, pos_tags)), total=len(flat_token_ids)):
        if pos == "NUM":
            num_token_infos.append((i, tok, pos))
            tok_freqs[tok] += 1

    # Show the number of NUM tokens found and some examples. 
    # Note that they are often not whole words. If you print some of the less frequent ones, you'll see they're not even numbers! (Spacy limitation)
    print("Aantal NUM-tokens gevonden:", len(num_token_infos))
    print("Voorbeelden:", num_token_infos[:5])
    for tok, freq in sorted(tok_freqs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{freq:>6}: {tokenizer.decode([tok])}")

    assert keys.shape[0] == len(vals), f"{keys.shape[0]} != {len(vals)}, keys and vals should have the same length"
    print("Shape van keys:", keys.shape)
    print("Aantal vals:", len(vals))

    # Get the hidden states and token_ids to save. Keep them as *torch tensors*
    num_indices = [i for i, _, _ in num_token_infos]
    num_keys = keys[num_indices]  # torch tensor
    num_token_ids = vals[num_indices]

    print("Tokens extracted:")
    print([tokenizer.decode(tok) for tok in num_token_ids[:20]])
    print(f"Chosen tokens from datastore are same as extracted NUM tokens: {set(num_token_ids.flatten().tolist()) == set(tok_freqs.keys())}")

if SAVE:
    raw_data_file_name = f"num_data_{DATASET}.pkl"
    with open(os.path.join(ROOT, DATASET, raw_data_file_name), "wb") as f:         # raw_ds_data_file name may change if using speaker_idxs or all hidden states
        pickle.dump({"hidden_states": num_keys,
                    "idxs": num_token_ids}, f)
