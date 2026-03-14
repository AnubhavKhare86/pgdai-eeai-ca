"""
translation.py
--------------
Optional pre-processing step that translates non-English ticket text
into English before the TF-IDF vectoriser runs.

The pipeline uses Facebook's M2M100 multilingual translation model
together with Stanza's language-identification processor.  If the
detected language is already English the text is passed through
unchanged.  A handful of Stanza language codes that M2M100 does not
recognise are remapped to the nearest supported code.

This step is opt-in: pass --use-translation on the command line to
activate it.  Without it the pipeline runs faster and the translation
dependencies (stanza, transformers) do not need to be installed.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer

# Stanza language codes that M2M100 does not support, mapped to the
# closest language it does support.
_LANG_REMAP = {
    'fro': 'fr',   # Old French -> French
    'la':  'it',   # Latin -> Italian (closest Romance language)
    'nn':  'no',   # Norwegian Nynorsk -> Norwegian Bokmal
    'kmr': 'tr',   # Kurmanji Kurdish -> Turkish
}

_MODEL_NAME = 'facebook/m2m100_418M'


def trans_to_en(texts: list) -> list:
    """
    Translate a list of strings to English.

    Each string is language-identified with Stanza.  If the detected
    language is English the original string is returned as-is.
    Otherwise the M2M100 model translates it to English.

    Empty strings are passed through without calling either model,
    which saves time when many rows have no Ticket Summary.

    Args:
        texts: List of raw text strings, possibly in mixed languages.

    Returns:
        A list of English strings in the same order as the input.
    """
    # Load the translation pipeline and tokenizer once for the whole batch.
    t2t_pipe  = pipeline(task='text2text-generation', model=_MODEL_NAME)
    model     = M2M100ForConditionalGeneration.from_pretrained(_MODEL_NAME)
    tokenizer = M2M100Tokenizer.from_pretrained(_MODEL_NAME)

    # Stanza multilingual pipeline for language identification only.
    nlp_stanza = stanza.Pipeline(
        lang='multilingual',
        processors='langid',
        download_method=DownloadMethod.REUSE_RESOURCES,
    )

    translated = []
    for text in texts:
        # Skip empty strings - nothing to translate.
        if text == '':
            translated.append(text)
            continue

        doc  = nlp_stanza(text)
        lang = doc.lang

        if lang == 'en':
            # Already English - no translation needed.
            translated.append(text)
        else:
            # Remap unsupported language codes before passing to M2M100.
            lang = _LANG_REMAP.get(lang, lang)

            tokenizer.src_lang = lang
            encoded = tokenizer(text, return_tensors='pt')
            tokens  = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id('en'),
            )
            text_en = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            translated.append(text_en)

    return translated
