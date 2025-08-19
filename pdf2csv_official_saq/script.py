import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import fitz
import numpy as np
import cv2
import json
from fitz import Rect
import re
from collections import defaultdict
from typing import Dict, List
import argparse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def page_saver(project):
    pdf_path = "./pdf/" + project + ".pdf"
    doc = fitz.open(pdf_path)
    os.makedirs(f"./pages/{project}", exist_ok=True)
    for target_page_num in range(1, len(doc)):
        page = doc[target_page_num]
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = img.copy()
        cv2.imwrite(f"./pages/{project}/{target_page_num-1}.jpg", img)

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
    margin = 5
    for doc_idx in range(1, doc.page_count):
        page_idx = doc_idx - 1
        page = doc[doc_idx]
        bboxes = page.cluster_drawings(x_tolerance=30, y_tolerance=30)
        for i, bbox in enumerate(bboxes):
            expanded_bbox = Rect(
                bbox.x0 - margin,
                bbox.y0 - margin,
                bbox.x1 + margin,
                bbox.y1 + margin
            )
            pix = page.get_pixmap(clip=expanded_bbox)
            filename = f"./diagram/{project_name}/diagram_{page_idx}_{i}.png"
            pix.save(filename)

def get_topic_id_with_gpt(question_text, subject_id, client):
    topic_list = get_topic_list(subject_id)
    text = "The questions topic candidate like below {topic_name, topic_id}:"
    for topic in topic_list:
        topic_id = topic["topic_id"]
        topic_name = topic["topic_name"]
        text += f"\n{topic_name.split('(')[0]}: {topic_id}"
    respone = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that transcribes PDF files to text."},
            {"role": "user", "content": f"Please provide the topic_id for the question: {question_text}"},
            {"role": "user", "content": text},
            {"role": "user", "content": r"you respond in json format like {topic_id: integer_value}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(respone.choices[0].message.content)["topic_id"]

QUESTION_PATTERN = re.compile(r'^\s*(\d+)\.\s*(?:\t\s*)?$', re.MULTILINE)

def parse_question_pages(doc: str, page_token: str = "<PAGE_START>") -> Dict[int, List[int]]:
    pages_raw = doc.split(page_token)
    if pages_raw and pages_raw[0].strip() == "":
        pages_raw = pages_raw[1:]
    q_pages: Dict[int, List[int]] = defaultdict(list)
    current_q: int | None = None
    for page_idx, page_text in enumerate(pages_raw):
        m = QUESTION_PATTERN.search(page_text)
        if m:
            current_q = int(m.group(1))
        if current_q is None:
            continue
        q_pages[current_q].append(page_idx)
    return dict(q_pages)

def marksheme_diagram_saver(project_name):
    doc = fitz.open(f"./pdf/{project_name}_markscheme.pdf")
    margin = 15
    os.makedirs(f"./mark_scheme_diagram/{project_name}", exist_ok=True)
    for page_num in range(1, doc.page_count):
        page = doc[page_num]
        img_index_offset = 0
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
            filename = f"./mark_scheme_diagram/{project_name}/diagram_{page_num}_{img_index + img_index_offset}.png"
            cv2.imwrite(filename, cropped_img)

def answer_parser_vision(project_name, client):
    doc = fitz.open(f"./pdf/{project_name}_markscheme.pdf")
    marksheme_diagram_saver(project_name)
    diagram_list = os.listdir(f"./mark_scheme_diagram/{project_name}")
    diagram_list.sort(key=lambda x: int(x.split("_")[1]))
    encoded_page_list = []
    encoded_diagram_dict = {}
    for diagram_name in diagram_list:
        if diagram_name.startswith(f"diagram_"):
            diagram_path = f"./mark_scheme_diagram/{project_name}/{diagram_name}"
            encoded_diagram = encode_image(diagram_path)
            encoded_diagram_dict[diagram_path] = encoded_diagram
    answer_dict = {}
    for page_num in range(1, doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = img.copy()
        cv2.imwrite(f"./comp.jpg", img)
        encoded_entire_image = encode_image(f"./comp.jpg")
        encoded_page_list.append(encoded_entire_image)
    content = [{"type": "text", "text": 'I need question_number:answer dictionary. you must reference the image below. And return {question_number:mark_scheme_text} in json format. mark_scheme_text must be generated by merging answers element with notes element. you do not need to notes key individually. question number is generated by 1-(a), 1-(b)-(i), 1-(b)-(ii), 2-(a)-(i) format. If the question number is empty, this means pre-question-number is continued. in this case you can reference full page image.'},
                {"type": "text", "text": "mark scheme can be equations, so if mark scheme is equations, you must write them using LaTeX syntax according to the following rules." },
                {"type": "text", "text": "Change pure text to LaTeX syntax (This is really really really important rule you must follow this instructuion): - All mathematical expressions and numbers must be enclosed in $...$. - Plain text should remain outside of $...$. Thus, you should never wrap entire sentences in LaTeX. - Use $\\newline$ whenever there is a new line. - If the transcriber is missing $ where it should be (around any LaTeX commands or mathematical expressions), make sure to wrap it correctly. It is crucial to enclose all math-related content within $...$ - For numbers with units, wrap the number in $...$, and use \\textrm{{}} for the unit, placing a ~ (space) between them. Example: $5 \\textrm{{~kg}}$ - If there is a percentage sign (%) in a mathematical equation, DO NOT use a backslash (\\) before it. Leave it as it is."},
                {"type": "text", "text": "Please convert the following content into KaTeX-compatible format. All mathematical expressions should be wrapped in either \\( ... \\) for inline math or $$ ... $$ for display math. Avoid using \n for line breaks—use double spaces (for markdown) or <br> if HTML is intended. Ensure all math functions (like \frac, \Omega, \text, etc.) are properly enclosed in math mode, and separate plain text from math expressions clearly."},
                {"type": "text", "text": "This is the whole concatenated image of the PDF page. You can see overview by referring to this image."},
                {"type": "text", "text": "The following image is the PDF individual page containing the problem you need to digitize. As you can see, it includes a mixture of equations and diagrams."},
                {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(f'./full_page/{project_name}.jpg')}"}}, 
                {"type": "text", "text": "The following image is the PDF individual page containing the problem you need to digitize. As you can see, it includes a mixture of equations and diagrams."},]
    for idx, encoded_entire_image in enumerate(encoded_page_list):
        content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_entire_image}"}})
    content.append({"type": "text", "text": f"Mark scheme can have diagrams in answer or notes. So I share diagrams with you below. You must insert in mark_scheme_text in right place with [img: image_path] tag. image path will be provided by me. "})
    for key, value in encoded_diagram_dict.items():
        content.append({"type": "text", "text": f"Below image path is {key}"})
        content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{value}"}})
    content.append({"type": "text", "text": "Please provide the answer in JSON format."})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that transcribes PDF files to text."},
            {"role": "user", "content": content}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    indiv_answer_dict = json.loads(response.choices[0].message.content)
    answer_dict.update(indiv_answer_dict)
    return answer_dict

def answer_parser_refiner(answer_dict, client):
    with open("raw_answer.json", "r", encoding="utf-8") as f:
        answer_dict = json.load(f)
    refined_answer_dict = {}
    system_prompt = "You are an expert in KaTeX and Markdown rendering. Convert invalid or unsafe KaTeX into safe Markdown using inline math ($...$) only. Do not use $$...$$ or unsupported environments. Output must be JSON with the same key and a cleaned-up string value."
    for key, value in answer_dict.items():
        user_prompt = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": f"Refine this key-value pair into safe KaTeX JSON:\n{{\"{key}\": \"{value}\"}}"},
            {"type": "text", "text": "Do not omit the key. Do not alter diagram formats like [img: image_path]."}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            refined = json.loads(content)
            refined_answer_dict.update(refined)
        except Exception as e:
            print(f"Failed to process {key}: {e}")
            refined_answer_dict[key] = value
    with open("refined_answer.json", "w", encoding="utf-8") as f:
        json.dump(refined_answer_dict, f, indent=2, ensure_ascii=False)
    return refined_answer_dict

def get_page2diagram_dict(project_name):
    page2diagram_dict = {}
    diagram_list = os.listdir(f"./diagram/{project_name}")
    diagram_list.sort(key=lambda x: int(x.split("_")[1]))
    diagram_list.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    for diagram_name in diagram_list:
        if diagram_name.startswith(f"diagram_"):
            page_num = int(diagram_name.split("_")[1])
            if page_num not in page2diagram_dict:
                page2diagram_dict[page_num] = []
            page2diagram_dict[page_num].append(diagram_name)
    return page2diagram_dict

def full_page_saver(project_name):
    doc = fitz.open(f"./pdf/{project_name}_markscheme.pdf")
    comp_img = None
    for page_num in range(2, doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = img.copy()
        if comp_img is None:
            comp_img = img
        else:
            comp_img = np.concatenate((comp_img, img), axis=0)
    cv2.imwrite(f"./full_page/{project_name}.jpg", comp_img)

def main():
    parser = argparse.ArgumentParser(description="Process problem set PDF and mark scheme PDF to generate per-question JSON objects.")
    parser.add_argument("--project_name", required=True, help="Project name (e.g., Mathematics_N24-TZ0-P1(HL)-*)")
    parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., 6)")
    args = parser.parse_args()
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    project_name = args.project_name
    subject_id = args.subject_id
    full_page_saver(project_name)
    pdf_path = f"./pdf/{project_name}.pdf"
    doc = fitz.open(pdf_path)
    page_saver(project_name)
    pdf_text = ""
    for page_num in range(1, doc.page_count):
        page = doc[page_num]
        pdf_text += "<PAGE_START>\n"
        pdf_text += page.get_text()
        pdf_text += "\n"
    pdf_text = pdf_text.replace("", "").replace("\x08", "").replace("\xa02", "").replace("\xa0", "").replace("\u2009", "")
    problem_page_dict = parse_question_pages(pdf_text)
    # First get the raw answer dictionary from vision parsing
    raw_answer_dict = answer_parser_vision(project_name, client)
    
    # Save the raw answer for debugging purposes
    with open("raw_answer.json", "w", encoding="utf-8") as f:
        json.dump(raw_answer_dict, f, indent=2, ensure_ascii=False)
    
    # Then refine the answers
    saq_answer_dict = answer_parser_refiner(raw_answer_dict, client)
    print("Parse answer dict successfully")
    os.makedirs(f"./diagram/{project_name}", exist_ok=True)
    diagram_saver(project_name)
    for k in list(problem_page_dict.keys()):
        mark_scheme_per_problem = ""
        for q_num in saq_answer_dict.keys():
            if q_num.startswith(f"{k}-"):
                mark_scheme_per_problem += f"{q_num}: {saq_answer_dict[q_num]}\n"
        content = [{"type": "text", "text": """다음 텍스트에서 문제 구조를 분석하여 재귀적인 딕셔너리 트리로 파싱해주세요. 출력은 JSON 형식입니다.

아래의 파싱 규칙을 반드시 지켜주세요:

[질문 유형 구분]
1. 각 질문은 반드시 다음 셋 중 하나의 타입을 가집니다:
   - main-question: 번호로 시작하는 최상위 질문 (예: 1., 2., 3.)
   - sub-question: 괄호로 시작하는 질문 (예: (a), (b), (i), (ii), ...)
   - context: 해당 질문에 앞서 위치하며, 문제 해결에 필요한 전제나 조건을 제공하는 설명 문장

2. sub-question은 중첩될 수 있습니다.  
   예: (c)(i), (c)(ii)는 (c)의 하위 sub-question입니다.
              
3. question_text는 문제의 본문을 나타내며, 반드시 question number 1. 2. (a), (b), (c), (i), (ii) 등을 포함하고 total_mark인 [숫자]([1],[2] 등)를 포함해선 안됩니다.
    - 예: "question_text": "1. Find the value of x."
    - 예: "question_text": "(a) A car travels at a speed of 60 km/h."
    - 예: "question_text": "(ii) draw a diagram to show the forces acting on the car."

4. context는 항상 **바로 아래의 질문(sub-question or main-question)에 속하는 전제**로 간주합니다.  
   즉, context는 질문보다 먼저 나와야 하며, 뒤에 이어지는 질문의 내용에 영향을 줍니다.

5. 질문 내부의 점수 정보는 [3], [2] 등으로 제공되며 total_mark 필드로 추출되어야 합니다.

6. topic_id는 해당 문제가 속하는 주제의 id를 나타냅니다. 아래 주어진 {topic_id: topic} 딕셔너리를 보고 알맞는 topic_id를 할당해주세요.
              
""" + making_topic_tagging_prompt(get_topic_list(subject_id)) + """
7. mark_scheme_text는 문제의 채점 기준을 나타내며, 아래 첨부되어 있습니다. total_mark가 없다면 mark_scheme_text도 null로 설정되어야 합니다.
""" + mark_scheme_per_problem + """
8. use_diagram은 학생이 해당 문제를 위해 그림을 그려서 제출해야 할 것 같은지 주관적인 판단을 한 후 넣어주세요. Draw a diagram 등이 있다면 그림이 필요할 것입니다. True/False로 설정해주세요. 통계적으로 use_diagram을 사용하는 문제는 그다지 많지 않습니다. 강한 확신이 없다면 False로 설정해주세요.

위 조건을 모두 만족시킬 수 있도록 파싱해주세요.
""" + """
[출력 구조]
- 출력은 다음과 같은 재귀적 딕셔너리 형태를 가집니다:
  {
    "1": {
      "question_type": "main-question",
      "question_text": "...",
      "mark_scheme_text": "...",
      "total_mark": null,
      "topic_id": "...",
      "use_diagram": "...",
      "sub": {
        "a": {
          "question_type": "sub-question",
          "question_text": "...",
          "mark_scheme_text": "...",
          "total_mark": null,
          "topic_id": "...",
          "use_diagram": "...",
          "sub": {
            "i": {
              "question_type": "sub-question",
              "question_text": "...",
              "mark_scheme_text": "...",
              "total_mark": 3,
              "topic_id": "...",
              "use_diagram": "...",
              "sub": {}
            }
          }
        },
        "context1": {
          "question_type": "context",
          "question_text": "...",
          "mark_scheme_text": "...",
          "total_mark": null,
          "topic_id": "...",
          "use_diagram": "...",
          "sub": {
            ...
          }
        }
      }
    }
  }

[주의할 점 – 실수 방지를 위한 조건]
- 질문 지시자((a), (i) 등)가 중첩되는 경우에는 반드시 상위-하위 관계로 정리해야 합니다.
- context 문장은 question_text로 포함시키지 말고 별도 context 블록으로 분리하세요.
- context는 항상 질문보다 먼저 나와야 하며, 질문과 논리적으로 연결되는지 판단해서 배속하세요.
- total_mark는 질문 본문 끝의 [숫자] 형식으로부터 추출하세요.
- 필요하다면 context에 번호를 붙이되 (예: "context1", "context2"), 한 문제 안에서 유일하게 유지하세요.
- root question과 sub-question 사이에는 절대 context가 올 수 없습니다.
              
[중요한점]
- dictionary를 만들기 전에 전반적인 문제 구조를 이해하고, 각 질문의 관계를 파악하세요.
- sub-question을 갖지 않는 context는 절대 존재할 수 없습니다. context는 항상 sub-question을 가집니다.

이 기준에 맞게 입력 텍스트를 손실 없이 파싱해 주세요."""}]
        content.append({"type": "text", "text": f"Question number is {k}"})
        page_list = problem_page_dict[k]
        base64_page_dict = {}
        for page_num in page_list:
            base64_page_dict[page_num] = encode_image(f"./pages/{project_name}/{page_num}.jpg")
            content.append({"type": "text", "text": f"This is the image of page you must refrence for responding to my request"})
            content.append({"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_page_dict[page_num]}"}})
        content.append({"type": "text", "text": "The Question Text and Options can include diagrams, and you must include these diagrams by embedding the image names within the text of the Question Text and Options according to the following rules. Rewrite question_text and option: an image is included, insert the [img: image_url], [img: image_url] tag metioned above image at the appropriate location in the text. - It is very important to insert the tag at the correct location. Also, the Question Text and Options might not be regular text but mathematical expressions. In such cases, you must write them using LaTeX syntax according to the following rules." \
                        + "Change pure text to LaTeX syntax: \
                            - All mathematical expressions and numbers must be enclosed in $...$. \
                            - Plain text should remain outside of $...$. Thus, you should never wrap entire sentences in LaTeX. \
                            - Use $\\newline$ whenever there is a new line. - If the transcriber is missing $ where it should be (around any LaTeX commands or mathematical expressions), make sure to wrap it correctly. It is crucial to enclose all math-related content within $...$. \
                            - For numbers with units, wrap the number in $...$, and use \\textrm{{}} for the unit, placing a '~' (space) between them. Example: $5 \\textrm{{~kg}}$ - If there is a percentage sign (%) in a mathematical equation, DO NOT use a backslash (\\) before it. Leave it as it is."})
        for page_num in page_list:
            for idx, img_url in enumerate(get_page2diagram_dict(project_name).get(page_num, [])):
                content.append({"type": "text", "text": f"Below image is ./diagram/{project_name}/{img_url} you must insert. if image is not meaningful, you must drop it. please drop the not meaningful image this is important!!! if barcode and answer box looking like multiline pleaseplease. For example, barcode image is not meaningful and answer box you write is not meaningful. Only diagram student reference is meaningful."})
                content.append({"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(f'./diagram/{project_name}/{img_url}')}"}})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that transcribes PDF files to text."},
                {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        response_dict = json.loads(response.choices[0].message.content)
        os.makedirs(f"./{project_name}/json_folder", exist_ok=True)
        with open(f"./{project_name}/json_folder/{project_name.split('*')[0]+str(k)+project_name.split('*')[-1]}.json", "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=4)

if __name__ == "__main__":
    main()
