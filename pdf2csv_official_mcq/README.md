# PDF MCQ to CSV/Excel Converter

This project provides a script to automate the digitization of multiple-choice question (MCQ) problem sets from PDF files, including answer extraction, diagram/image handling, topic tagging, and LaTeX formatting. The script leverages OpenAI's LLMs and vision APIs to process both the question set and its corresponding mark scheme, outputting a structured Excel file suitable for further use.

## Features

- **PDF Parsing:** Extracts questions and choices from MCQ PDFs using regular expressions.
- **Answer Extraction:** Parses answers from the mark scheme PDF, with optional vision-based extraction for complex cases.
- **Diagram/Image Handling:** Extracts diagrams from questions and uploads them to S3, replacing image tags in the text with cloud URLs.
- **LLM Integration:** Uses OpenAI's GPT-4o for reasoning tasks, such as image insertion, topic tagging, and LaTeX formatting.
- **Topic Tagging:** Associates each question with a topic ID, using a provided topics.json.
- **Output:** Generates a structured Excel file with all questions, options, answers, and metadata.

## Input Files

- **Problem Set PDF:** e.g., `pdf/Physics_M22-TZ2-P1-*(HL).pdf`
- **Mark Scheme PDF:** e.g., `pdf/Physics_M22-TZ2-P1-*(HL)_markscheme.pdf`
  - The mark scheme filename must match the problem set filename, with `_markscheme` appended before `.pdf`.
- **topics.json:** Topic mapping for each subject.
- **subject2id.json:** Mapping from subject name to subject ID.
- Problem source in excel column is fully dependent on input file names. 
- Recommend to name pdf like (Subject_name)_(Problem_source_Format + "-*" + .pdf) form. * character is replaced with problem number automatically. (e.g Physics_M22-TZ2-P1-*(HL).pdf)

## Usage

```bash
python script.py --project "PROJECT_NAME" --subject_id SUBJECT_ID
```

- `PROJECT_NAME`: The base name of the problem set PDF (without extension), e.g., `Physics_M22-TZ2-P1-*(HL)`
- `SUBJECT_ID`: (Optional) The subject ID. If not provided, it is inferred from the project name using `subject2id.json`.

- Naming Rule is very important. You should includ -* in pdf name like below example
- And must include "" to project name

### Example

```bash
python ./script.py --project "Physics_M22-TZ2-P1-*(HL)"
```

## How It Works

1. **Argument Parsing:**  
   The script uses `argparse` to accept `--project` and optional `--subject_id` arguments. All logic is contained within a `main()` function, executed only if the script is run directly.

2. **Directory Setup:**  
   Output directories for diagrams, pages, JSON, and Excel are created based on the project name.

3. **PDF Processing:**  
   - The problem set PDF is opened and each page is saved as an image.
   - Diagrams are extracted from each page and saved as separate images.

4. **Question Parsing:**  
   - Questions and choices are extracted from the PDF text using regular expressions (`parse_questions_with_choices`).
   - Each question is associated with its page number.

5. **Answer Extraction:**  
   - Answers are parsed from the mark scheme PDF (`answer_parser`).
   - Optionally, vision-based extraction (`answer_parser_vision`) can be used for complex mark schemes.

6. **LLM Reasoning:**  
   - For each question, a prompt is constructed including the question text, options, diagrams, and candidate topics.
   - The prompt is sent to OpenAI's GPT-4o, which returns a JSON with the digitized question, options, and topic ID, formatted with LaTeX and image tags as needed.

7. **Image Upload and Tag Replacement:**  
   - Any image tags in the question/options are replaced with S3 URLs after uploading the corresponding images.

8. **Output Generation:**  
   - Each processed question is saved as a JSON file.
   - All questions are aggregated and saved as an Excel file in the `excel/` directory.

## Output

- **Excel File:**  
  `excel/{PROJECT_NAME}.xlsx` containing all digitized questions, options, answers, and metadata.

- **JSON Files:**  
  `json_folder/{PROJECT_NAME}/*.json` for each question.

- **Extracted Images:**  
  `diagram/{PROJECT_NAME}/diagram_{page_num}_{img_index}.png`

- **Page Images:**  
  `pages/{PROJECT_NAME}/{page_num}.jpg`

## Environment Variables

Set the following in your `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-northeast-2
```

## Code Reference

- **Argument Handling:**  
  See the `main()` function and `argparse` usage at the bottom of `script.py`.
- **PDF and Image Processing:**  
  Functions: `page_saver`, `diagram_saver`, `encode_image`
- **Question/Answer Parsing:**  
  Functions: `parse_questions_with_choices`, `answer_parser`, `answer_parser_vision`
- **LLM Prompting and Processing:**  
  Functions: `get_prompt`, `gpt_execute`
- **Image Upload and Tag Replacement:**  
  Functions: `img_upload`, `replace_img_tag`
- **Excel Output:**  
  Functions: `save_to_excel`, `load_json_files_from_folder`

## Notes

- The script is designed for batch processing of IB-style MCQ PDFs and their mark schemes.
- All logic is contained under `if __name__ == "__main__":` for safe import and modularity.
- Project and subject IDs are always provided via command-line arguments, never as hardcoded variables.

## License

MIT License
