"""A tiny original char-level corpus (about the book's own subject) so the lab needs no download."""
CORPUS = """\
A transformer block reads a stream of vectors and writes small corrections back into it.
The attention layer lets each token look at the tokens before it and copy what it needs.
The feed-forward layer then thinks about each token alone, using the same weights for every position.
Residual connections add the input back to the output, so information can flow straight through the stack.
Layer normalization keeps the numbers in a friendly range so that training stays stable.
A language model is trained to guess the next character, and then the next, and then the next.
When it guesses wrong, the loss goes up, and gradient descent nudges every weight a little.
After many nudges the model has learned that a q is usually followed by a u, and that spaces follow words.
Small models learn spelling first, then short words, then the rhythm of whole sentences.
Large models learn grammar, facts, style, and even a little reasoning, all from the same simple game.
The decoder only design hides the future with a causal mask, so each position can only see its past.
The encoder only design lets every token see every other token, which suits understanding tasks.
The encoder decoder design reads the whole input first and then writes the answer one token at a time.
A good engineer counts parameters before training, because memory and cost follow from that number.
The embedding table is one big matrix, and each row is the vector for one token in the vocabulary.
Tying the input embedding to the output layer saves memory and often helps small models.
Doubling the width of the model makes the feed-forward layers four times larger.
Stacking more layers gives the model more steps of computation before it must answer.
More heads let the model watch several patterns at once, each in a smaller subspace.
The idea is simple: read, attend, think, add, and repeat, layer after layer, until the answer appears.
Pre normalization places the norm inside the residual branch, which keeps a clean path for gradients.
Post normalization places it after the addition, which was the original design but is harder to train deep.
Modern models often use a root mean square norm and a gated activation in the feed-forward layer.
Sampling with a low temperature picks safe characters, while a high temperature makes wilder choices.
A tiny model trained for one minute will babble, but the babble already looks like language.
Every large language model began as a small experiment very much like this one.
"""

# --- Templated extension -----------------------------------------------------------------------------
# 2.5 KB is far too small: a model memorises it in seconds and validation loss explodes. We add ~30 KB of
# grammatical sentences generated from fixed word lists with a seeded RNG. It is a toy language with real
# structure to learn (word order, agreement, punctuation), so validation loss genuinely falls.
import random as _random

_ADJ = ["small", "large", "careful", "quiet", "fast", "patient", "curious", "tired", "bright", "simple"]
_NOUN = ["model", "engineer", "layer", "token", "network", "student", "server", "dataset", "head", "vector"]
_VERB = ["reads", "writes", "checks", "updates", "finds", "copies", "ignores", "trains", "predicts", "stores"]
_ADV = ["slowly", "quickly", "carefully", "again", "today", "twice", "first", "later"]
_PLACE = ["in the cloud", "on the laptop", "at the lab", "in the stack", "on the cluster"]


def _sentences(n, seed=0):
    r = _random.Random(seed)
    out = []
    for _ in range(n):
        a1, n1, v, a2, n2 = r.choice(_ADJ), r.choice(_NOUN), r.choice(_VERB), r.choice(_ADJ), r.choice(_NOUN)
        form = r.randrange(4)
        if form == 0:
            out.append(f"The {a1} {n1} {v} the {a2} {n2} {r.choice(_ADV)}.")
        elif form == 1:
            out.append(f"A {a1} {n1} {v} a {a2} {n2} {r.choice(_PLACE)}.")
        elif form == 2:
            out.append(f"When the {n1} {v} the {n2}, the {a2} {r.choice(_NOUN)} {r.choice(_VERB)} it.")
        else:
            out.append(f"Every {a1} {n1} {r.choice(_VERB)} {r.choice(['one', 'two', 'three'])} {a2} {n2}s {r.choice(_ADV)}.")
    return " ".join(out)


CORPUS = CORPUS + "\n" + _sentences(450)
