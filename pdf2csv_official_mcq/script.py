import base64
import os
import argparse
from openai import OpenAI
import fitz
import numpy as np
import cv2
import json
from fitz import Rect
import tqdm
import re
import boto3
from openpyxl import Workbook
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_questions_with_choices(text):
    pattern = r'(?m)^(\d+)\.\s*\n'
    matches = list(re.finditer(pattern, text))

    questions = {}

    for i in range(len(matches)):
        number = matches[i].group(1)
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[start:end].strip()

        choice_pattern = r'(A[\.\s]+.*?)(?=B[\.\s]+|$)'
        a_match = re.search(choice_pattern, block, re.DOTALL)
        if not a_match:
            continue

        question_part = block[:a_match.start()].strip()
        choices_text = block[a_match.start():]

        options = re.findall(r'([A-D][\.\s]+.*?)(?=[A-D][\.\s]+|$)', choices_text, re.DOTALL)

        options = [
            re.sub(r'^([A-D])\.\s+', r'\1 ', opt.strip()) 
            for opt in options
        ]

        questions[number] = {
            'question': question_part,
            'options': options
        }

    return questions

def page2text(page): #doc[page_num]
    texts = page.get_text()
    return texts

def page_saver(project):
    pdf_path = "./pdf/" + project + ".pdf"
    doc = fitz.open(pdf_path)
    os.makedirs(f"./pages/{project}", exist_ok=True)

    for target_page_num in range(len(doc)):
        page = doc[target_page_num]
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        img = img.copy()
        cv2.imwrite(f"./pages/{project}/{target_page_num}.jpg", img)

def is_question_number(text: str) -> bool:
    txt = text.strip()
    return txt.endswith('.') and txt[:-1].isdigit()

def answer_parser(project_name):    
    doc = fitz.open(f"./pdf/{project_name}_markscheme.pdf")
    page = doc[-1]
    text = page.get_text()

    lines = text.splitlines()
    
    answers = {}
    
    question_number = None
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\.$', line):
            question_number = int(line.rstrip('.'))
        else:
            if question_number is not None:
                if line in ['A','B','C','D']:
                    answers[question_number] = line
                question_number = None
    
    return answers

def answer_parser_vision(project_name):
    doc = fitz.open(f"./pdf/{project_name}_markscheme.pdf")
    page = doc[-1]
    pix = page.get_pixmap()
    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img_array.reshape(pix.h, pix.w, pix.n)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = img.copy()
    cv2.imwrite(f"./answer.jpg", img)

    encoded_entire_image = encode_image(f"./answer.jpg")

    content = [{"type": "text", "text": 'I need question_number:answer dictionary. you must reference the image below. And return {question_number:answer} in json format. question answer should be A,B,C,D. if question_answer is -, you must ignore this question.'},
            {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_entire_image}"}
                        }]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that transcribes PDF files to text."},
            {"role": "user", "content": content}
        ],
        response_format= { "type": "json_object" },
        temperature=0.0
        )
    answer_dict = json.loads(response.choices[0].message.content)
    answer_dict = {int(k): v for k, v in answer_dict.items()}
    return answer_dict

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

def diagram_saver(project_name):
    doc = fitz.open(f"./pdf/{project_name}.pdf")
    margin = 15

    for page_num in range(2, doc.page_count):
        page = doc[page_num]
        bboxes = page.cluster_drawings(x_tolerance = 30, y_tolerance = 30)

        img_index_offset = 0
        for i, bbox in enumerate(bboxes):
            expanded_bbox = Rect(
                bbox.x0 - margin,
                bbox.y0 - margin,
                bbox.x1 + margin,
                bbox.y1 + margin
            )

            pix = page.get_pixmap(clip=expanded_bbox)
            filename = f"./diagram/{project_name}/diagram_{page_num}_{i}.png"
            pix.save(filename)
            
            img_index_offset += 1

        images = page.get_images(full=True)
        pix = page.get_pixmap()

        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            bbox = page.get_image_rects(xref)[0]
            x0, y0, x1, y1 = map(int, [bbox.x0, bbox.y0, bbox.x1, bbox.y1])
            cropped_img = img[y0:y1, x0:x1] 
            filename = f"./diagram/{project_name}/diagram_{page_num}_{img_index + img_index_offset}.png"
            cv2.imwrite(filename, cropped_img) 

def get_problem_graph(project_name):
    doc = fitz.open(f"./pdf/{project_name}.pdf")
    problem_graph = {}
    for page_num in range(1, doc.page_count):
        problem_dict = parse_questions_with_choices(page2text(doc[page_num]))

        for key in problem_dict.keys():
            problem_dict[key]['page_num'] = page_num
        
        problem_graph.update(problem_dict)

    return problem_graph

def extract_image_numbers(text):
    pattern = r'\[img:\s*([^\]]+)\]'
    matches = re.findall(pattern, text)
    return matches

def img_upload(local_path, img_name):
    S3_BUCKET_NAME = "ibguru"

    try:
        s3_url = f"https://{S3_BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/{img_name}.png"
        s3_client.upload_file(local_path, S3_BUCKET_NAME, f"{img_name}.png", ExtraArgs={"ContentType": "image/png"})
        return s3_url
    except:
        return local_path.split("/")[-1].split(".")[0]
    

def replace_img_tag(text, img_name, cloud_path):
    pattern = rf'\[img:\s*{re.escape(img_name)}\s*\]'  
    replaced_text = re.sub(pattern, f'[{cloud_path}]', text)
    return replaced_text

def get_prompt(project_name, question_text, options, page_num, subject_id):
    encoded_entire_image = encode_image(f"./pages/{project_name}/{page_num}.jpg")
    diagram_path = []

    all_diagram_list = os.listdir(f"./diagram/{project_name}")

    for diagram_name in all_diagram_list:
        if diagram_name.startswith(f"diagram_{page_num}"):
            diagram_path.append(f"./diagram/{project_name}/{diagram_name}")

    encoded_diagram = [encode_image(diagram) for diagram in diagram_path]

    options = "\n".join([f"{opt}" for opt in options])
    content = [{"type": "text", "text": "You must follow my instruction. I need your help, I am currently automating the process of digitizing problems, and your role is to assist me in this digitization process. \
                Digitization is completed when you fill in the following six values and return them in JSON format. The dictionary you need to fill in looks like this. \
                {question_text: '', topic_id: '', option1: '', option2:'', option3: '', option4:''} The question_text of the problem that needs to be digitized is as follows:"\
                 + question_text + "The options for the problem that needs to be computerized are as follows:Option1 is A, Option2 is B, Option3 is C, and Option4 is D. You must include A,B,C,D in option1,2,3,4 text" \
                    + options + "Do not add . at end of the option number don't follow A. blabla format must follow A blabla format."}]
    content.append({"type": "text", "text": "The Question Text and Options can include diagrams, and you must include these diagrams by embedding the image names within the text of the Question Text and Options according to the following rules. Rewrite question_text and option: an image is included, insert the [img: image1], [img: image2] tag metioned above image at the appropriate location in the text. - It is very important to insert the tag at the correct location. Also, the Question Text and Options might not be regular text but mathematical expressions. In such cases, you must write them using LaTeX syntax according to the following rules." \
                    + "Change pure text to LaTeX syntax: \
                        - All mathematical expressions and numbers must be enclosed in $...$. \
                        - Plain text should remain outside of $...$. Thus, you should never wrap entire sentences in LaTeX. \
                        - Use $\\newline$ whenever there is a new line. - If the transcriber is missing $ where it should be (around any LaTeX commands or mathematical expressions), make sure to wrap it correctly. It is crucial to enclose all math-related content within $...$. \
                        - For numbers with units, wrap the number in $...$, and use \\textrm{{{{}}}} for the unit, placing a '~' (space) between them. Example: $5 \\textrm{{~kg}}$ - If there is a percentage sign (%) in a mathematical equation, DO NOT use a backslash (\\) before it. Leave it as it is."})
    content.append({"type": "text", "text": "The following image is the entire PDF page containing the problem you need to digitize. As you can see, it includes a mixture of equations and diagrams."})
    content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_entire_image}"}
                        },)
    content.append({"type": "text", "text": f"Image index start in 0 ex)image0, image1, image2."})
    for i in range(len(encoded_diagram)):
        content.append({"type": "text", "text": f"This is image{i}."})
        content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_diagram[i]}"}})
        
    content.append({"type":"text", "text": f"Fill in topic_id: - You will be provided with candidate topics. Choose the most appropriate one for each question. Candidate is like below: - You only fill in the topic_id field in the main-question. {making_topic_tagging_prompt(get_topic_list(subject_id))} - topic id must be numbers, do not add any other characters or sentences"})
    content.append({"type":"text", "text": f"you must end the json and sentence do not add token infinitly. this is important. do not add token infinitly !"})
    return content

def gpt_execute(project_name, problem_question, problem_options, page_num, subject_id):
    prompt = get_prompt(
            project_name,
            problem_question,
            problem_options,
            page_num,
            subject_id
        )
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that transcribes PDF files to text."},
        {"role": "user", "content": prompt}
    ],
    response_format= { "type": "json_object" },
    temperature=0.0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {}

def get_problem_attribute(topic_id, source, question_text, option1, option2, option3, option4, correct_option):
    translate_alphabet2num={
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4
    }
    
    return {
        "generate": True,
        "is_subquestion": False,
        "parent_id": "",
        "topic_id": topic_id,
        "type": "MCQ",
        "source": source,
        "question_text": question_text,
        "total_mark": 1,
        "attachment_file": "",
        "option1": option1,
        "option2": option2,
        "option3": option3,
        "option4": option4,
        "correct_option": translate_alphabet2num[correct_option],
        "correct_explanation": "",
        "incorrect_explanation": "",
        "link": "",
        "use_math_input": True,
        "use_diagram": False,
        "requires_working_out": True,
        "correct_answer": "",
        "explantion": "N/A",
        "mark_scheme": "",
        "pdf_uri": "",
        "transcript": "",
        "essay_mark_scheme": "",
        "tags": ""
    }

def extract_number(filename: str) -> int:
    parts = filename.split('-')
    number = parts[-1]
    return int(number.split('.')[0].split("(")[0])

def load_json_files_from_folder(folder_path: str):
    json_data_list = []

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')],
        key=lambda f: os.path.getctime(os.path.join(folder_path, f))
    )
    for filename in files:
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    json_data = json.load(file)
                    json_data_list.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")
    return json_data_list

def sanitize_value(value):
    try:
        value = value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    except:
        pass
    illegal_xml_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
    if isinstance(value, str):
        value = illegal_xml_chars.sub('', value)
        return value
    return value

def save_to_excel(json_data_list, output_file: str):
    if not json_data_list:
        print("No data to save.")
        return

    workbook = Workbook()
    sheet = workbook.active

    column_names = list(json_data_list[0].keys())
    sheet.append(column_names)

    for json_data in json_data_list:
        row = [sanitize_value(json_data.get(key, '')) for key in column_names]
        sheet.append(row)

    workbook.save(output_file)
    print(f"Data saved to {output_file}")

def subject2id(subject):
    subject_dict = json.load(open("subject2id.json", "r"))
    return subject_dict[subject]

def main():
    parser = argparse.ArgumentParser(description="Parse MCQ PDFs and generate digitized CSV/Excel output.")
    parser.add_argument("--project", required=True, help="Project name (e.g., Chemistry_M25-SPEC-P1a-*)")
    parser.add_argument("--subject_id", required=False, help="Subject ID (if not provided, derived from project name)")
    args = parser.parse_args()

    project_name = args.project
    if args.subject_id:
        subject_id = args.subject_id
    else:
        subject_id = subject2id(project_name.split("_")[0])

    pdf_path = f"./pdf/{project_name}.pdf"
    diagram_folder_path = f"./diagram/{project_name}"

    os.makedirs(diagram_folder_path, exist_ok=True)
    os.makedirs(f"./pages/{project_name}", exist_ok=True)
    os.makedirs(f"./json_folder/{project_name}", exist_ok=True)
    os.makedirs("excel", exist_ok=True)

    doc = fitz.open(pdf_path)
    page_saver(project_name)
    diagram_saver(project_name)
    problem_graph = get_problem_graph(project_name)

    answer_dict = answer_parser(project_name)
    # answer_dict = answer_parser_vision(project_name)

    keys = list(problem_graph.keys())
    keys.sort(key=lambda x: int(x))

    for key in tqdm.tqdm(keys):
        # Extract the base pattern and replace * with question number
        # For project names like "Physics_paper_1__HL", we need to find the part with *
        # If no * exists, create a proper source identifier
        if "*" in project_name:
            problem_src = project_name.replace("*", key)
        else:
            # Generate source based on project pattern
            problem_src = f"{project_name.split('_')[0]}_Q{key}"

        if os.path.exists(f"./json_folder/{project_name}/{problem_src}.json"):
            print(f"File already exists for question {key}. Skipping...")
            continue

        question_text = problem_graph[key]['question']
        options = problem_graph[key]['options']
        page_num = problem_graph[key]['page_num']
        result = gpt_execute(project_name, question_text, options, page_num, subject_id)

        if not result:
            print(f"Error: No result for question {key}.")
            continue

        for valid_object in ["question_text", "option1", "option2", "option3", "option4"]:
            img_list = extract_image_numbers(result[valid_object])

            for image_tag in img_list:
                img_id = image_tag[5:]
                image_name = f"{problem_src}_{img_id}"
                img_path = f"./diagram/{project_name}/diagram_{page_num}_{img_id}.png"
                cloud_path = img_upload(img_path, image_name)
                replaced_text = replace_img_tag(result[valid_object], image_tag, cloud_path)
                result[valid_object] = replaced_text

        problem_attribute = get_problem_attribute(
            topic_id=result["topic_id"],
            source=problem_src,
            question_text=result["question_text"],
            option1=result["option1"],
            option2=result["option2"],
            option3=result["option3"],
            option4=result["option4"],
            correct_option=answer_dict[int(key)]
        )

        with open(f"./json_folder/{project_name}/{problem_src}.json", "w") as json_file:
            json.dump(problem_attribute, json_file, indent=4)

    folder_path = f"./json_folder/{project_name}/"
    output_file = f"./excel/{project_name}.xlsx"

    json_data_list = load_json_files_from_folder(folder_path)
    save_to_excel(json_data_list, output_file)

if __name__ == "__main__":
    main()
