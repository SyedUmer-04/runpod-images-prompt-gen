import runpod
from transformers import pipeline
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/mistral-v0.3")

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="float16",
    device_map="auto",
)

def handler(job):
    job_input = job["input"]
    word = job_input["word"]
    category = job_input["category"]
    country = job_input["country"]

    prompt_text = (
        f"<s>[INST] "
        f"Create a detailed image generation prompt for representing {word} in the context "
        f"of {category}."
        f"The image is for an educational flashcard for school students. The subject must "
        f"be immediately recognizable and guessable by a child at first glance. "
        f"Use {country} context ONLY when the subject is something that genuinely looks different or unique in {country} compared to other countries — such as traditional clothing, cultural items, local food and cuisine, religious symbols, government or military uniforms, architecture, or nationally specific objects. "
        f"For universal everyday items — such as body parts, furniture, school supplies, suitcases, electronics, vehicles, or common household objects — depict them in their standard, globally recognizable form, since they look the same everywhere including {country}. "
        f"Do NOT force {country} flag colors, patterns, or cultural styling onto neutral universal objects. Ask yourself: would this object look noticeably different in {country} vs. any other country? If no — draw it universally. If yes — apply {country} context. "
        f"The prompt must follow these rules: "
        f"- Pure white background, subject isolated cleanly with no scene or environment. "
        f"- Single focused subject only, no clutter. "
        f"- Bright, clear, saturated colours friendly for children. "
        f"- No text, labels, letters or numbers anywhere in the image. "
        f"- Detailed and descriptive — cover shape, colour, texture, style, angle, lighting. "
        f"- Flat front-facing or three-quarter view for maximum clarity. "
        f"- Illustration style, clean edges, even bright lighting. "
        f"- No shadows, no backgrounds, no extra objects unless part of the subject itself. "
        f"Output only the image generation prompt, nothing else. No explanation, no preamble. "
        f"[/INST]"
    )

    result = pipe(prompt_text, max_new_tokens=1000)[0]["generated_text"]
    result_clean = result.replace(prompt_text, "").strip()

    return {
        "value": word,
        "prompt": result_clean
    }

runpod.serverless.start({"handler": handler})
