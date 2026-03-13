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
    countryContext = job_input["countryContext"]

    prompt_text = (
        f"<s>[INST] "
        f"Create a detailed image generation prompt for representing {word} in the context "
        f"of {category}, specifically focused for {country}. "
        f"The image is for an educational flashcard for school students. The subject must "
        f"be immediately recognizable and guessable by a child at first glance. "
        f"Use the following country context to ensure every visual detail — appearance, "
        f"colour, style, shape, and form — is accurate to how this word genuinely looks "
        f"and exists in that country: {countryContext} "
        f"The prompt must follow these rules: "
        f"- Pure white background, subject isolated cleanly with no scene or environment. "
        f"- Single focused subject only, no clutter. "
        f"- Bright, clear, saturated colours friendly for children. "
        f"- No text, labels, letters or numbers anywhere in the image. "
        f"- Detailed and descriptive — cover shape, colour, texture, style, angle, lighting. "
        f"- Flat front-facing or three-quarter view for maximum clarity. "
        f"- Illustration style, clean edges, even bright lighting. "
        f"- No shadows, no backgrounds, no extra objects unless part of the subject itself. "
        f"Output only the image generation prompt, nothing else. No explanation, no preamble. and Don't Include Pakistani Flag or colors in everything, only where it is necessary!"
        f"[/INST]"
    )

    result = pipe(prompt_text, max_new_tokens=1000)[0]["generated_text"]
    result_clean = result.replace(prompt_text, "").strip()

    return {
        "value": word,
        "prompt": result_clean
    }

runpod.serverless.start({"handler": handler})
