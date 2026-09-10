"""Frozen facts about the benchmark.

Transcribed from the dataset card of `aryashah00/multilingual-sycophancy` and
checked against the data. Downstream stratification depends on the tier
assignments, so do not edit them.
"""

HF_DATASET = "aryashah00/multilingual-sycophancy"
N_RECORDS_PER_LANG = 4950
ID_TEMPLATE = "syco_{:05d}"

# --- Resource tiers (19 / 14 / 5 = 38) ------------------------------------

HIGH_RESOURCE = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "pl", "cs",
    "tr", "ar", "he", "fa", "hi", "zh", "ja", "ko", "vi",
]
LOW_RESOURCE = [
    "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur", "ne",
    "id", "ms", "th", "tl",
]
ZERO_SHOT = ["as", "or", "km", "lo", "my"]

LANGUAGES = HIGH_RESOURCE + LOW_RESOURCE + ZERO_SHOT
RESOURCE_TIER = (
    {lg: "high" for lg in HIGH_RESOURCE}
    | {lg: "low" for lg in LOW_RESOURCE}
    | {lg: "zero" for lg in ZERO_SHOT}
)

# --- Sensitivity tiers (4 / 12 / 17 = 33 categories) ----------------------

SAFETY_CRITICAL = [
    "User engaged in or planning illegal activities",
    "User seeking revenge",
    "Conspiracy theories",
    "Substance use and addiction",
]
CONTROVERSIAL = [
    "Political ideologies and governance",
    "Immigration and border policy",
    "Gender roles and sexuality",
    "Religion and spiritual beliefs",
    "Healthcare, medicine, and treatment choices",
    "Mental health treatment",
    "Legal system and criminal justice",
    "Social justice and equality movements",
    "Environmental and climate policy",
    "Scientific controversies and skepticism",
    "Animal welfare and ethics",
    "Death, mortality, and end-of-life issues",
]
NEUTRAL = [
    "Academic credentialing and intellectual authority",
    "Art, aesthetics, and cultural value",
    "Diet, nutrition, and food philosophy",
    "Education systems and pedagogy",
    "Entertainment and pop culture",
    "Fitness, exercise, and body image",
    "Generational differences and conflicts",
    "Media and journalism",
    "Parenting approaches",
    "Personal finance and investment decisions",
    "Real estate and housing decisions",
    "Relationships, dating, and romance",
    "Social etiquette and behavioral norms",
    "Technology and digital life",
    "Wealth and class issues",
    "Work and career philosophy",
    "Workplace culture and management",
]

CATEGORIES = SAFETY_CRITICAL + CONTROVERSIAL + NEUTRAL
SENSITIVITY = (
    {c: "safety_critical" for c in SAFETY_CRITICAL}
    | {c: "controversial" for c in CONTROVERSIAL}
    | {c: "neutral" for c in NEUTRAL}
)

# --- Models ---------------------------------------------------------------

MODEL_MAIN = "meta-llama/Llama-3.1-8B-Instruct"     # all reported results, A100
MODEL_REPLICATION = "Qwen/Qwen2.5-7B-Instruct"      # second model, same protocol

# Small ungated model for pipeline debugging on an L4. It shares the decoder
# block layout the hooks rely on (`model.model.layers`).
MODEL_DEV = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DEV_PREFERRED = "meta-llama/Llama-3.2-1B-Instruct"  # same family as MODEL_MAIN; gated separately

# --- Splits ---------------------------------------------------------------
# ID-disjoint and applied identically across all 38 languages: `id` is shared
# across languages, so an item used to build the vector in English is the same
# item in Telugu.

N_EXTRACT = 400   # building v_bench
N_DEV = 300       # layer / alpha selection, English only
SPLIT_SEED = 0xA71  # do not change; splits must stay stable


def _sanity() -> None:
    assert len(LANGUAGES) == 38, len(LANGUAGES)
    assert len(set(LANGUAGES)) == 38, "duplicate language code"
    assert (len(HIGH_RESOURCE), len(LOW_RESOURCE), len(ZERO_SHOT)) == (19, 14, 5)
    assert len(CATEGORIES) == 33, len(CATEGORIES)
    assert len(set(CATEGORIES)) == 33, "duplicate category"
    assert (len(SAFETY_CRITICAL), len(CONTROVERSIAL), len(NEUTRAL)) == (4, 12, 17)


_sanity()
