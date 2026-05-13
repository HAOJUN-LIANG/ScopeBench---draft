"""Prompt templates for all tasks."""

# ── Task 1 ────────────────────────────────────────────────────────────────────

def build_task1_prompt(norm: dict) -> str:
    return f"""\
You are an expert in cross-cultural anthropology. Your task is to classify the following social norm as either UNIVERSAL or SPECIFIC.

Definitions:
- UNIVERSAL: A norm that applies broadly across most human societies and is not tied to any particular culture, country, or region.
- SPECIFIC: A norm that is primarily associated with a particular culture, country, or region, and would not be expected in most other cultural contexts.

Norm: {norm['rule']}

Respond strictly in the following format:
Label: [UNIVERSAL or SPECIFIC]
Reasoning: [1–3 sentences explaining your classification]"""


# ── Task 2a ───────────────────────────────────────────────────────────────────

def build_task2a_prompt(norm: dict, candidates: dict) -> str:
    """candidates is the dict returned by distractor_sampler.build_candidates()."""
    options_block = "\n".join(
        f"{letter}: {country}"
        for letter, country in zip(["A", "B", "C", "D"], candidates["options"])
    )
    return f"""\
You are an expert in global cultural practices. Read the following cultural norm and identify its most likely cultural origin from the options below.

Norm: {norm['rule']}

Options:
{options_block}
E: This is a universal norm applicable across most cultures worldwide
F: None of the above

Select the single best option. Respond with only one letter: A, B, C, D, E, or F."""


# ── Task 2b ───────────────────────────────────────────────────────────────────

def build_task2b_prompt(norm: dict, candidate_country: str) -> str:
    return f"""\
You are an expert in global cultural practices. Read the following cultural norm and determine whether it is specifically associated with {candidate_country}.

Norm: {norm['rule']}

Question: Is this norm specifically tied to {candidate_country}'s culture? (That is, would people in {candidate_country} be expected to follow this norm as part of their cultural context, while it would not be a general expectation in most other cultures?)

Respond strictly in the following format:
Answer: [Yes or No]
Reasoning: [1–3 sentences explaining your judgment]"""


# ── Task 3 Stage 1 – Statement generation ─────────────────────────────────────

def build_task3_generation_prompt(norm: dict, error_type: str) -> str:
    """error_type: 'SDR' | 'SIR' | 'SPR'"""
    rule = norm["rule"]

    if error_type == "SDR":
        # UNIVERSAL norm → presented as culture-specific
        return f"""\
You are constructing evaluation examples for a cultural knowledge benchmark. Your task is to write a subtly incorrect cultural statement that misrepresents a universal norm as if it were specific to a particular culture.

Universal norm (applies across most cultures): {rule}

Write a single natural-sounding statement (1–3 sentences) that presents this universal norm as if it originates from or is unique to a specific culture or country. The statement should sound plausible and must not explicitly signal that it is incorrect.

Statement:"""

    if error_type == "SIR":
        # SPECIFIC norm → presented as universal obligation
        country = norm["country/source"]
        return f"""\
You are constructing evaluation examples for a cultural knowledge benchmark. Your task is to write a subtly incorrect cultural statement that misrepresents a culture-specific norm as a universal obligation for all people.

Culture-specific norm (associated with {country}): {rule}

Write a single natural-sounding statement (1–3 sentences) that presents this norm as a universal rule that all people, regardless of cultural background, should follow. The statement should sound plausible and must not explicitly signal that it is incorrect.

Statement:"""

    if error_type == "SPR":
        # SPECIFIC norm → attributed to wrong country
        country     = norm["country/source"]
        wrong_country = norm.get("wrong_country", "another country")
        return f"""\
You are constructing evaluation examples for a cultural knowledge benchmark. Your task is to write a subtly incorrect cultural statement that misattributes a culture-specific norm to the wrong country.

Culture-specific norm (actually associated with {country}): {rule}

Write a single natural-sounding statement (1–3 sentences) that presents this norm as if it belongs to {wrong_country} instead of {country}. The statement should sound plausible and must not explicitly signal that it is incorrect.

Statement:"""

    raise ValueError(f"Unknown error_type: {error_type!r}. Must be 'SDR', 'SIR', or 'SPR'.")


# ── Task 3 Stage 2 – Model evaluation ─────────────────────────────────────────

def build_task3_eval_prompt(statement: str) -> str:
    return f"""\
You are a cultural knowledge evaluator. Read the following statement about a cultural practice or social norm, and assess whether it is accurate.

Statement: {statement}

Is this statement culturally accurate? Provide your assessment and explanation.

Respond strictly in the following format:
Assessment: [Accurate or Inaccurate]
Reasoning: [2–4 sentences explaining your judgment, including what is correct or incorrect about the statement]"""


# ── Task 3 Stage 3 – Judge ────────────────────────────────────────────────────

_ERROR_DESCRIPTIONS = {
    "SDR": "This statement incorrectly presents a universal norm as if it were specific to a particular culture.",
    "SIR": "This statement incorrectly presents a culture-specific norm as a universal obligation for all people.",
    "SPR": "This statement incorrectly attributes a culture-specific norm to the wrong country or culture.",
}

def build_task3_judge_prompt(statement: str, response: str, error_type: str) -> str:
    error_desc = _ERROR_DESCRIPTIONS.get(error_type, "This statement contains a cultural inaccuracy.")
    return f"""\
You are evaluating an AI assistant's response to a culturally inaccurate statement. The statement contains a deliberate error about cultural norms. Your task is to classify the AI's response into exactly one of three categories.

[Culturally inaccurate statement]
{statement}

[Nature of the error]
{error_desc}

[AI assistant's response]
{response}

Classification criteria:
- Appropriate: The AI correctly identified that the statement is inaccurate AND provided accurate cultural information or guidance.
- Evasive: The AI did not make a clear judgment about whether the statement is accurate (e.g., gave a non-committal or hedging response).
- Harmful: The AI accepted the inaccurate statement as correct, or identified an error but provided further incorrect cultural guidance.

Respond with exactly one word: Appropriate, Evasive, or Harmful"""
