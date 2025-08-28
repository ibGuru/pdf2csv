import json
import pandas as pd
import re
import boto3
import tqdm
import os
from dotenv import load_dotenv
import argparse

def img_upload(local_path, project, id_counter, img_counter, s3_client):
    S3_BUCKET_NAME = "ibguru"
    try:
        img_name = f"{project}_{id_counter}_{img_counter}_v4.jpg"
        s3_client.upload_file(local_path, S3_BUCKET_NAME, img_name)
        s3_url = f"https://{S3_BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/{img_name}"
        return s3_url
    except:
        return local_path.split("/")[-1].split(".")[0]

def replace_img_tags(text, project, id_counter, img_counter, s3_client):
    pattern = r"\[img:\s*(.*?)\]"
    def replacer(match):
        local_path = match.group(1)
        return f"[{img_upload(local_path, project, id_counter, img_counter[0], s3_client)}]"
    return re.sub(pattern, replacer, text)

def text_preprocessor(text, project, id_counter, img_counter, s3_client):
    if text is None:
        return text
    if text == "":
        return ""
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    modified_text = replace_img_tags(text, project, id_counter, img_counter, s3_client)
    return modified_text

def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
    return text

def json_to_excel(project_name, s3_client):
    excel_rows = []
    id_counter = [2]
    img_counter = [0]
    def read_problem(problem_data, parent_id, topic_id, source):
        question_type = problem_data["question_type"]
        question_id = id_counter[0]
        if question_type == "main-question":
            parent_id_val = ""
            is_sub_question = False
            topic_id_val = int(problem_data["topic_id"])
        else:
            parent_id_val = parent_id
            is_sub_question = True
            topic_id_val = int(topic_id)
        generate = True
        type_ = "SHORT_ANSWER"
        question_text = text_preprocessor(problem_data["question_text"], project_name, id_counter[0], img_counter, s3_client)
        if len(problem_data["sub"].keys()) != 0:
            total_mark = None
        else:
            total_mark = str(problem_data["total_mark"])
        attachment_file = ""
        option1 = ""
        option2 = ""
        option3 = ""
        option4 = ""
        correct_option = ""
        correct_explanation = ""
        incorrect_explanation = ""
        link = ""
        use_math_input = True
        use_diagram = True if problem_data["use_diagram"] == "true" or problem_data["use_diagram"] == "True" else False
        requires_working_out = True
        correct_answer = ""
        explantion = "N/A"
        mark_scheme = text_preprocessor(problem_data["mark_scheme_text"], project_name, id_counter[0], img_counter, s3_client)
        pdf_uri = ""
        transcript = ""
        essay_mark_scheme = ""
        tags = ""
        if total_mark == "None":
            total_mark = ""
        excel_rows.append([
            generate,
            is_sub_question,
            parent_id_val,
            topic_id_val,
            type_,
            source,
            question_text,
            total_mark,
            attachment_file,
            option1,
            option2,
            option3,
            option4,
            correct_option,
            correct_explanation,
            incorrect_explanation,
            link,
            use_math_input,
            use_diagram,
            requires_working_out,
            correct_answer,
            explantion,
            mark_scheme,
            pdf_uri,
            transcript,
            essay_mark_scheme,
            tags,
        ])
        id_counter[0] += 1
        for sub_problem in problem_data["sub"].keys():
            read_problem(problem_data["sub"][sub_problem], question_id, topic_id_val, source)
    
    json_folder = sorted([f for f in os.listdir(f"./{project_name}/json_folder/") if f.endswith('.json')],
        key=lambda f: os.path.getctime(os.path.join(f"./{project_name}/json_folder/", f))
    )
    for json_file in tqdm.tqdm(json_folder):
        source = json_file.split("_")[-1].split(".")[0]
        json_path = f"./{project_name}/json_folder/{json_file}"
        with open(json_path, "r") as f:
            problem_data = json.load(f)
        problem_data = problem_data[list(problem_data.keys())[0]]
        read_problem(problem_data, "", "", source)
    columns = [
        "generate",
        "is_subquestion",
        "parent_id",
        "topic_id",
        "type",
        "source",
        "question_text",
        "total_mark",
        "attachment_file",
        "option1",
        "option2",
        "option3",
        "option4",
        "correct_option",
        "correct_explanation",
        "incorrect_explanation",
        "link",
        "use_math_input",
        "use_diagram",
        "requires_working_out",
        "correct_answer",
        "explantion",
        "mark_scheme",
        "pdf_uri",
        "transcript",
        "essay_mark_scheme",
        "tags",
    ]
    df = pd.DataFrame(excel_rows, columns=columns)
    df = df.applymap(clean_text)
    excel_filename = f"{project_name}v7.xlsx"
    df.to_excel(excel_filename, index=False)

def main():
    parser = argparse.ArgumentParser(description="Convert per-question JSON objects to Excel and upload images to S3.")
    parser.add_argument("--project_name", required=True, help="Project name (e.g., Mathematics_N24-TZ0-P1(HL)-*)")
    args = parser.parse_args()
    load_dotenv()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    json_to_excel(args.project_name, s3_client)

if __name__ == "__main__":
    main()
