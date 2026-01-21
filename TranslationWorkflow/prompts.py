from langchain_core.prompts import PromptTemplate

TRANSLATE_TEMPLATE = """
You are an expert translator. Translate the following text from {source_language} to {target_language}.
Provide only the translated text, without any additional explanations or preambles.

Original Text:
{text}
"""

QUALITY_CHECK_PROMPT = """You are a strict translation quality reviewer.
    Review the initial translation based on the original text.
    Does the translation accurately convey the meaning, style, and nuances of the original?
    Answer with a single word: "Good" if the translation is excellent and needs no changes, or "Bad" if it has any issues or could be improved.

    Original Text ({source_language}):
    {original_text}

    Initial Translation ({target_language}):
    {initial_translation}

    Your single-word assessment:
    """

REFLECTION_TEMPLATE = """You are a senior translation reviewer. Your task is to review a translation based on the original text.
    Identify any potential issues in the translation regarding fluency, accuracy, terminology, and cultural nuances.
    Provide a concise list of constructive feedback and suggestions for improvement.

    Original Text ({source_language}):
    {original_text}

    Initial Translation ({target_language}):
    {initial_translation}

    Your Reflection and Suggestions:
    """

REFINE_TEMPLATE = """You are a master translator responsible for producing the final version of a translation.
    Use the original text, the initial translation, and the reviewer's reflection to create a polished and high-quality final translation.
    Integrate the suggestions from the reflection to improve upon the initial version.
    Provide only the final, refined translated text.

    Original Text ({source_language}):
    {original_text}

    Initial Translation ({target_language}):
    {initial_translation}

    Reviewer's Reflection and Suggestions:
    {reflection}

    Final Polished Translation:
    """

