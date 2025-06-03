import re
import json
from collections import OrderedDict
import fitz
import os
import cv2
import numpy as np

root_regex = r"^\s*([MN]\d{2}-TZ[012]-P[123]-(?:G\d+|\d{1,2})(?:\((SL|HL)\))?(?:\.\.\.)?)\s*$"

ROOT_PATTERN = re.compile(
    root_regex,
    re.MULTILINE
)

SUB_PATTERN = re.compile(
    r"\(([a-hj-z])\)\s+",
    flags=re.MULTILINE
)
SUB_SUB_PATTERN = re.compile(
    r"\((i{1,3}|iv|v?i{0,3}|ix|x)\)\s+",
    flags=re.MULTILINE
)
SCORE_PATTERN = re.compile(r"\[(\d+)\]") 

def clean_text4json(text):
    if text == None:
        return text
    else:
        text = text.split("Topic")[0]
        return text.replace("\n","")

def parse_exam(text):
    root_blocks = split_by_root_questions(text)
    
    results = {}
    for root_label, root_text_block in root_blocks:
        parsed_sub = parse_sub_questions(root_text_block)

        if parsed_sub:
            sub_positions = list(SUB_PATTERN.finditer(root_text_block))
            if sub_positions:
                first_sub_start_idx = min(m.start() for m in sub_positions)
                root_text_part = root_text_block[:first_sub_start_idx].strip()
            else:
                root_text_part = root_text_block.strip()

            total_score = sum_of_scores(parsed_sub.values())
            root_score = str(total_score) if total_score is not None else None

            root_data = {
                "question_type": "main-question",
                "question_text": clean_text4json(root_text_part),
                "mark_scheme_text": None,
                "total_mark": None,
                "topic_id": None,
                "use_diagram": False,
                "sub": parsed_sub
            }
        else:
            root_text, root_score, root_solution = split_score_solution(root_text_block.strip())
            root_data = {
                "question_type": "main-question",
                "question_text": clean_text4json(root_text),
                "mark_scheme_text": clean_text4json(root_solution),
                "total_mark": root_score,
                "topic_id": None,
                "use_diagram": False,
                "sub": parsed_sub
            }

        if root_label in results:
            dict_key = root_label + "@"
            while dict_key in results:
                dict_key += "@"
            results[dict_key.replace("\u200b","")] = root_data
        else:
            results[root_label.replace("\u200b","")] = root_data
    return results

def split_by_root_questions(text):
    matches = list(ROOT_PATTERN.finditer(text))
    if not matches:
        return []

    root_blocks = []
    for idx, match in enumerate(matches):
        root_label = match.group(1).strip()
        start_idx = match.end()  

        if idx < len(matches) - 1:
            end_idx = matches[idx + 1].start()
        else:
            end_idx = len(text)
        
        chunk = text[start_idx:end_idx]
        root_blocks.append((root_label.replace("\u200b",""), chunk))
    return root_blocks

def get_topic_list(subject_id):
    topic_dict = json.load(open("topics.json", "r"))
    return topic_dict[subject_id]

def parse_sub_questions(question_text):
    sub_matches = list(SUB_PATTERN.finditer(question_text))
    if not sub_matches:
        return {} 

    sub_dict = {}
    for idx, m in enumerate(sub_matches):
        label = m.group(1)  
        start_pos = m.end()
        if idx < len(sub_matches) - 1:
            end_pos = sub_matches[idx+1].start()
        else:
            end_pos = len(question_text)
        
        sub_chunk = question_text[start_pos:end_pos]
        sub_sub_parsed = parse_sub_sub_questions(sub_chunk)
        
        if sub_sub_parsed:
            sub_sub_positions = list(SUB_SUB_PATTERN.finditer(sub_chunk))
            if sub_sub_positions:
                first_sub_sub_start_idx = min(x.start() for x in sub_sub_positions)
                sub_text = sub_chunk[:first_sub_sub_start_idx].strip()
            else:
                sub_text = sub_chunk.strip()

            total_sub_sub_score = sum_of_scores(sub_sub_parsed.values())
            sub_score = str(total_sub_sub_score) if total_sub_sub_score is not None else None

            sub_dict[label] = {
                "question_type": "sub-question",
                "question_text": clean_text4json(f'({label}) ' + sub_text),
                "mark_scheme_text": None,
                "total_mark": None,
                "topic_id": None,
                "use_diagram": False,
                "sub": sub_sub_parsed
            }

        else:
            leaf_text, leaf_score, leaf_solution = split_score_solution(sub_chunk.strip())
            sub_dict[label] = {
                "question_type": "sub-question",
                "question_text": clean_text4json(f'({label}) ' + leaf_text),
                "mark_scheme_text": clean_text4json(leaf_solution),
                "total_mark": leaf_score,
                "topic_id": None,
                "use_diagram": False,
                "sub": {}
            }

    return sub_dict

def parse_sub_sub_questions(question_text):
    matches = list(SUB_SUB_PATTERN.finditer(question_text))
    if not matches:
        return {}

    sub_sub_dict = {}
    for idx, m in enumerate(matches):
        label = m.group(1).lower()
        start_pos = m.end()
        if idx < len(matches) - 1:
            end_pos = matches[idx+1].start()
        else:
            end_pos = len(question_text)
        
        chunk = question_text[start_pos:end_pos].strip()

        leaf_text, leaf_score, leaf_solution = split_score_solution(chunk)
        sub_sub_dict[label] = {
            "question_type": "sub-sub-question",
            "question_text": clean_text4json(f'({label}) ' + leaf_text),
            "mark_scheme_text": clean_text4json(leaf_solution),
            "total_mark": leaf_score,
            "topic_id": None,
            "use_diagram": False,
            "sub": {}
        }
    return sub_sub_dict

def split_score_solution(text_block):
    match = SCORE_PATTERN.search(text_block)
    if match:
        score_str = match.group(1)
        score_start = match.start()
        score_end = match.end()

        q_text = text_block[:score_start].strip()
        sol_text = text_block[score_end:].strip()

        return q_text, score_str, sol_text
    else:
        return text_block, None, None

def sum_of_scores(question_items):
    total = 0
    has_any_score = False
    for item in question_items:
        sc = item.get("score")
        if sc is not None:
            has_any_score = True
            try:
                total += int(sc)
            except ValueError:
                pass
    return total if has_any_score else None

def is_src(string):
    pattern = root_regex
    return bool(re.fullmatch(pattern, string))

def get_problem_page(pdf_name):
    doc = fitz.open(pdf_name)

    problem_graph = OrderedDict()
    processed_problem = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        texts = page.get_text()
        candidate_list = texts.split("\n")

        for element in candidate_list:
            element = element.rstrip().lstrip()
            if is_src(element):
                if processed_problem == element:
                    if processed_problem in problem_graph: #collision
                        while True:
                            if processed_problem in problem_graph:
                                processed_problem += "@"
                            else:
                                break
                    problem_graph[processed_problem.replace("\u200b","")] = [page_num]

                else:
                    if processed_problem != "":
                        problem_graph[processed_problem.replace("\u200b","")].append(page_num)
                    processed_problem = element
            
                    if processed_problem in problem_graph: #collision
                        while True:
                            if processed_problem in problem_graph:
                                processed_problem += "@"
                            else:
                                break
                    problem_graph[processed_problem.replace("\u200b","")] = [page_num]


    for problem in list(problem_graph.keys()):
        if len(problem_graph[problem]) == 2:
            start, end = problem_graph[problem][0], problem_graph[problem][1]
            problem_graph[problem] = list(range(start, end + 1))

    return problem_graph

def page_saver(project):
    pdf_path = "./pdf/" + project + ".pdf"
    doc = fitz.open(pdf_path)
    os.makedirs(f"./{project}/pages", exist_ok=True)

    for target_page_num in range(len(doc)):
        page = doc[target_page_num]
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_array.reshape(pix.h, pix.w, pix.n)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        img = img.copy()
        cv2.imwrite(f"./{project}/pages/{target_page_num}.jpg", img)

def validate_questions_structure(questions_dict: dict) -> bool:
    for key, question_data in questions_dict.items():
        if not _check_question_node(question_data):
            return False
    return True


def _check_question_node(question_node: dict) -> bool:
    mark_scheme = question_node.get("mark_scheme_text", None)
    sub_questions = question_node.get("sub", {})

    if mark_scheme is not None:
        return True
    
    if not sub_questions:
        return False

    for _, child_question in sub_questions.items():
        if _check_question_node(child_question):
            return True

    return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess exam PDF to generate problem graph JSON.")
    parser.add_argument("--project", required=True, help="Project name (used as PDF filename and output folder)")
    args = parser.parse_args()

    project = args.project
    pdf_name = project + ".pdf"
    pdf_path = "./pdf/" + pdf_name

    page_saver(project)

    doc = fitz.open(pdf_path)
    problem_page_dict = get_problem_page(pdf_path)

    with open(f'./test.json', 'w', encoding='utf-8') as f:
        json.dump(problem_page_dict, f, indent=2, ensure_ascii=False)

    sample_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        texts = page.get_text()
        page_num_digit = len(str(page_num+1))
        sample_text += texts[:-3-page_num_digit]
    
    result = parse_exam(sample_text)

    for dkey in list(result.keys()):
        try:
            result[dkey]["page"] = problem_page_dict[dkey]
        except:
            del result[dkey]

    with open(f'./{project}/problem_graph.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Total problems: ", len(result))
