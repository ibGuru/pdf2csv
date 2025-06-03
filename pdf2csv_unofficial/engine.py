import json
import fitz
import base64
import numpy as np
import cv2
from openai import OpenAI
import os
import tqdm
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def get_topic_list(subject_id):
    topic_dict = json.load(open("topics.json", "r"))
    return topic_dict[subject_id]

def making_topic_tagging_prompt(topic_list):
    text = "The questions topic candidate like below {topic_name, topic_id}:"
    for topic in topic_list:
        topic_id = topic["topic_id"]
        topic_name = topic["topic_name"]
        text += f"\n{topic_name.split('(')[0]}: {topic_id}"
    text += "\nPlease provide the topic_id for each question in json"
    return text

def save_decoded_image(encoded_string, output_path):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(encoded_string))

def image_logging(project,img_list, probelm_name):
    os.makedirs(f"./{project}/images/{probelm_name}", exist_ok=True)
    for encoded_image in img_list:
        save_decoded_image(encoded_image, f"./{project}/images/{probelm_name}/{img_list.index(encoded_image)+1}.jpg")

def make_prompt_message(project,subject_id, page_to_process, processed_probelm_src, text_for_page, image_for_page, full_page, json_to_process):

    image_logging(project,image_for_page, processed_probelm_src)
    
    content = [{"type": "text", "text": f'I want to convert the questions corresponding to {processed_probelm_src} from images to text. The {processed_probelm_src} question spans across the following two pictures.'}]
    for idx, page in enumerate(page_to_process):
        content.append({"type": "text", "text": f'This image shows the entirety of page {page+1}:'})
        content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{full_page[idx]}"}
                        },)
        content.append({"type": "text", 
                            "text": f'Page {page+1} consists of the following text: {text_for_page[idx]}'
                        },)
        
    for idx, encoded_image in enumerate(image_for_page):
        content.append({"type": "text", 
                            "text": f"image {idx+1} looks like this and is part of the pages I sent:" 
                        },
                        )
        content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"}
                        })
    content.append({"type": "text",
                        "text": f'Among images, there may be meaningless images. In such cases, boldly ignore those images. Randomly cropped images should be considered meaningless. Even if a part of the cropped image seems meaningful, if the image as a whole appears to be haphazardly cropped, it should be disregarded. However, images can also be graphs, and in the case of graphs, do not ignore them but preserve them properly, as graphs are not considered meaningless images. Do not include meaningless images in the image_tag according to the specified rules.'},)
    content.append({"type": "text",
                    "text": \
f"""I need you to process and digitize a given problem set {processed_probelm_src}.
                            
The digitization process follows these steps:
- You will be provided with a PDF image containing the problem set.
- You will also receive a structured JSON file that categorizes the problems.

Provided Json like below:
{json.dumps(json_to_process)}

Your task is to refine and complete the JSON file by doing the following:

Rewrite question_text and mark_scheme_text:
- If an image is included, insert the [img: image1], [img: image2] tag metioned above image at the appropriate location in the text. 
- It is very important to insert the tag at the correct location.
- These fields contain pure text but may include mathematical expressions.
- If there are mathematical expressions, rewrite them using LaTeX syntax.
- These fields may also contain images.
- If an image is an equation, convert it to LaTeX after OCR and add it to the string text this is very very important. use must follow this rule. this is very important.; 
- do not add it to the image_tag.
- You must not add problem src and total mark to the question text.
- Sometimes, question text and mark scheme text include chapter name like B2.2 Organelles and ~~ or 1. HL Section A Data Answer or problem source M17-TZ2~, this is not a part of the question text and mark scheme text, so you must remove it. Do not include it in the question text and mark scheme text.'

Change pure text to LaTeX syntax:
- All mathematical expressions and numbers must be enclosed in $...$.
- Plain text should remain outside of $...$. Thus, you should never wrap entire sentences in LaTeX.
- Use $\\newline$ whenever there is a new line.
- If the transcriber is missing $ where it should be (around any LaTeX commands or mathematical 
  expressions), make sure to wrap it correctly. It is crucial to enclose all math-related content 
  within $...$.
- For numbers with units, wrap the number in $...$, and use \\textrm{{{{}}}} for the unit, placing a "~" (space) between them.
  Example: $5 \\textrm{{~kg}}$
- If there is a percentage sign (%) in a mathematical equation, DO NOT use a backslash (\\) before it. 
  Leave it as it is.

Fill in topic_id:
- You will be provided with candidate topics. Choose the most appropriate one for each question.

Candidate is like below:
- You only fill in the topic_id field in the main-question.
{making_topic_tagging_prompt(get_topic_list(subject_id))}

Determine use_diagram:
- This requires your judgment.
- If solving the problem requires drawing a diagram, set it to true; otherwise, set it to false.
- A parent question having sub-questions or sub-sub-questions is always false.

You must always end the json. You must never increase the backslash infinitely.
"""
})
    
    content.append({"type": "text",
                     "text": f'Please provide the JSON format for the question {processed_probelm_src}. not string, i want the JSON format.'})
    return content

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_info(project,doc,page_list):
    text_for_page = []
    image_for_page = []
    full_page = []

    rejected_set = set(["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAACAdQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD6sooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//Z"])

    for page_num in page_list:
        page = doc[page_num]

        images = page.get_images(full=True)
        texts = page.get_text()
        pix = page.get_pixmap()

        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n) #img is page image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./{project}/pages/{page_num}.jpg", img)
        full_page.append(encode_image(f"./{project}/pages/{page_num}.jpg"))

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        text_for_page.append(texts)

        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            bbox = page.get_image_rects(xref)[0]
            x0, y0, x1, y1 = map(int, [bbox.x0, bbox.y0, bbox.x1, bbox.y1])
            cropped_img = img[y0:y1, x0:x1] 
            height, width = cropped_img.shape[:2]

            if height == 452 and width == 452:
                continue

            if height == 451 and width == 452:
                continue

            if height == 3 and width == 468:
                continue

            if height == 3 and width == 452:    
                continue

            if height == 0 or width == 0:
                continue

            random_path = f"./{project}/junk/" + str(uuid.uuid4()) + ".jpg"
            cv2.imwrite(random_path, cropped_img) #comp image is the image cropped from the pdf
            encoded_image = encode_image(random_path)
            if encoded_image in rejected_set:
                continue
            image_for_page.append(encoded_image)
            os.remove(random_path)

    return text_for_page, image_for_page, full_page

def get_message(project,subject_id,page_to_process, processed_probelm_src, text_for_page, image_for_page, full_page, json_to_process):
    messages = [{"role": "system", "content": "You are a helpful assistant that transcribing pdf file to text."},
            {"role": "user", "content": make_prompt_message(project,subject_id,page_to_process, processed_probelm_src, text_for_page, image_for_page, full_page,json_to_process)}]
    return messages

def process_json(project,problem_graph,subject_id,problem_name):
    if Path(f'./{project}/json_folder/{problem_name}.json').exists():
        return True
    
    json_to_process = problem_graph[problem_name]

    page_to_process = json_to_process["page"][:]
    del json_to_process["page"]
    text_for_page, image_for_page, full_page = get_info(project,fitz.open(f"./pdf/{project}.pdf"), page_to_process)
    messages = get_message(project,subject_id,page_to_process, problem_name, text_for_page, image_for_page, full_page,json_to_process)
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format= { "type": "json_object" },
                temperature=0.0,
            )

    response_text = response.choices[0].message.content
    json_string = response_text

    try:
        data = json.loads(json_string)
        with open(f'./{project}/json_folder/{problem_name}.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except:
        with open(f"./{project}/error_log.txt", "a") as f:
            f.write(f"{problem_name}\n")
            f.write(f"{json_string}\n")
        return False

def subject2id(subject):
    subject_dict = json.load(open("subject2id.json", "r"))
    return subject_dict[subject]

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process problem_graph.json and generate per-question JSONs using GPT.")
    parser.add_argument("--project", required=True, help="Project name (used as folder and file prefix)")
    parser.add_argument("--subject", required=True, help="Subject name (used to look up subject_id)")
    args = parser.parse_args()

    project = args.project
    subject = args.subject
    subject_id = subject2id(subject)

    os.makedirs(f"./{project}/images/", exist_ok=True)
    os.makedirs(f"./{project}/json_folder/", exist_ok=True)
    os.makedirs(f"./{project}/junk/", exist_ok=True)

    with open(f"./{project}/problem_graph.json", "r", encoding='utf-8') as f:
        problem_graph = json.load(f)

    problem_names = list(problem_graph.keys())

    max_workers = 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(process_json, project, problem_graph, subject_id, problem_name): problem_name
            for problem_name in problem_names
        }

        for future in tqdm.tqdm(as_completed(future_to_problem), total=len(problem_names)):
            problem_name = future_to_problem[future]
            try:
                result = future.result()
                if not result:
                    print(f"[Error Detected] {problem_name}")
            except Exception as e:
                print(f"[Exception] {problem_name}: {e}")

if __name__ == "__main__":
    main()
