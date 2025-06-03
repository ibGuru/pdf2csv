import json
import pandas as pd
import re
import boto3
import tqdm
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

id_counter = 2
img_counter = 0

def img_upload(local_path, project):
    global img_counter
    img_counter += 1
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

    try:
        img_name = f"{project}_{id_counter}_{img_counter}_v4.jpg"
        s3_client.upload_file(local_path, S3_BUCKET_NAME, img_name)
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{img_name}"
        return s3_url
    except:
        print("error detected!")
        return local_path.split("/")[-1].split(".")[0]

def text_preprocessor(problem_name, text, project):
    if text == None:
        return text
    
    if text == "":
        return ""

    text = text.replace("\n","\\n").replace("\t","\\t")

    pattern = r"\[img: (image\d+)\]"
    matches = re.findall(pattern, text)
    tag_dict = {}

    if len(matches) == 0:
        return text

    for idx,tag in enumerate(matches):
        img_id = tag.split("image")[1]
        cloud_path = img_upload(f"./{project}/images/{problem_name}/{img_id}.jpg", project)
        tag_dict[tag] = f"[{cloud_path}]"

    def replace_with_cloud_link(match):
        image_name = match.group(1)  # "image2" 같은 부분만 가져오기
        return tag_dict.get(image_name, match.group(0))
    
    modified_text = re.sub(pattern, replace_with_cloud_link, text)

    return modified_text

def clean_text(text):
    if isinstance(text, str):  # 문자열인지 확인
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)  # 제어 문자 제거
    return text

#making excel
def json_to_excel(project_name):
    with open(f"./{project_name}/problem_graph.json", "r") as f:
        problem_graph = json.load(f)

    excel_rows = []
    error_list = []

    def read_problem(problem_data, parent_id, topic_id):
        try:
            global id_counter

            question_type = problem_data["question_type"]
            question_id = id_counter

            if question_type == "main-question":
                parent_id = ""
                is_sub_question = False
                topic_id = int(problem_data["topic_id"])
            else:
                parent_id = parent_id
                is_sub_question = True
                topic_id = int(topic_id)

            generate = True
            type_ = "SHORT_ANSWER"

            source = problem_name.replace("@", " ")

            question_text = text_preprocessor(problem_name, problem_data["question_text"], project_name)

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
            use_diagram = problem_data["use_diagram"]
            requires_working_out = True
            correct_answer = ""
            explantion = "N/A"
            mark_scheme = text_preprocessor(problem_name, problem_data["mark_scheme_text"], project_name)
            pdf_uri = ""
            transcript = ""
            essay_mark_scheme = ""
            tags = ""

            if total_mark == "None":
                total_mark = ""

            excel_rows.append([
                generate,
                is_sub_question,
                # question_id,
                parent_id,
                topic_id,
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

            id_counter += 1

            for sub_problem in problem_data["sub"].keys():
                read_problem(problem_data["sub"][sub_problem], question_id, topic_id)
        except:
            print(f"Error! in {problem_name}")
            error_list.append(problem_name)
            pass

    problem_name_list = list(problem_graph.keys())

    for problem_name in tqdm.tqdm(problem_name_list):
        json_path = f"./{project_name}/json_folder/{problem_name}.json"
        try:
            with open(json_path, "r") as f:
                problem_data = json.load(f)
            read_problem(problem_data, "", "")
        except:
            print(f"Error! in {problem_name}")
            error_list.append(problem_name)
            pass

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

    with open(f"./{project_name}/parsing_error.txt", "w") as f:
        for item in error_list:
            f.write("%s\n" % item)

    for i in range(len(excel_rows)):
        if excel_rows[i][5] in error_list:
            excel_rows[i][0] = False

    df = pd.DataFrame(excel_rows, columns=columns)
    df = df.applymap(clean_text)
    # df.fillna("", inplace=True)
    excel_filename = f"{project_name}v5.xlsx"
    df.to_excel(excel_filename, index=False)

#making image link
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess per-question JSONs and generate Excel output.")
    parser.add_argument("--project", required=True, help="Project name (used as folder and file prefix)")
    args = parser.parse_args()

    project = args.project
    json_to_excel(project)
