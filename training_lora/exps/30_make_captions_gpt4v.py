import glob
import shutil
import os
import openai
from mimetypes import guess_type
import base64
import concurrent.futures
from tqdm import tqdm

system_prompt = """This is an image of Tohoku Zunko, a Japanese anime character.

・Tohoku Zunko is a Tohoku region character with a zunda (edamame) motif.
・Call her Zunko.
・Sometimes she has an edamame bow.

Create a prompt for Stable Diffusion. Please answer only the prompt.
"""

def create_caption(base64_encoded_image):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": base64_encoded_image,
                        "detail": "low"
                    }
                }]
            }
        ], 
        model='gpt-4-turbo', 
        temperature=0.2,
        max_tokens=500)
    gpt4v_output = response.choices[0].message.content

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Please convert to include zunko in all sentences."},
            {"role": "user", "content": gpt4v_output}
        ], 
        model='gpt-4-turbo', 
        temperature=0.2,
        max_tokens=500)
    replace_output = response.choices[0].message.content.lower()
    return replace_output

def make_dataset(source_png_file, dataset_root="zunko03"):
    target_png_path = source_png_file.replace("/zunko_orig/", f"/{dataset_root}/")
    shutil.copy2(source_png_file, target_png_path)

    with open(source_png_file, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    mime_type = guess_type(source_png_file)[0]
    data_url=f"data:{mime_type};base64,{base64_encoded_data}"

    captions = create_caption(data_url)
    target_txt_path = target_png_path.replace(".png", ".txt")
    with open(target_txt_path, "w", encoding="utf-8") as f:
        f.write(captions)

def main():
    os.makedirs("../data/zunko03", exist_ok=True)

    source_png_files = sorted([x.replace("\\", "/") for x in glob.glob("../data/zunko_orig/*.png")])

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(make_dataset, source_png_files)))
    
if __name__ == "__main__":
    main()