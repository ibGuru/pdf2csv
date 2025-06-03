# PDF Problem Book to Structured Data Pipeline

This project processes problem book PDFs into structured data (JSON and Excel) using a three-stage pipeline:

- **Preprocessor**: Extracts and analyzes question structure from PDF, generating a problem graph.
- **Engine**: Uses GPT to reason about and enrich the problem graph, producing detailed per-question JSONs.
- **Postprocessor**: Converts the enriched JSONs into a final Excel file, uploading images to S3 and updating references.

## Directory Structure

```
.
├── preprocessor.py
├── engine.py
├── postprocessor.py
├── subject2id.json
├── topics.json
├── pdf/
│   └── <project>.pdf
├── <project>/
│   ├── pages/
│   ├── images/
│   ├── json_folder/
│   └── problem_graph.json
└── ...
```

## 1. Preprocessor (`preprocessor.py`)

**Purpose:**  
Analyzes the PDF to detect the hierarchical structure of questions (root, sub, sub-sub), extracts question text, marks, and relationships, and generates `<project>/problem_graph.json`.

**How it works:**  
- Uses regular expressions to identify root questions, sub-questions, and sub-sub-questions.
- Extracts question text, marks, and page associations.
- Saves each PDF page as an image in `<project>/pages/`.
- Outputs a JSON graph describing the structure and content of all questions.

**Usage:**
```bash
python preprocessor.py --project <project>
```
- `<project>`: The base name of your PDF file (without `.pdf`). The PDF must be located at `./pdf/<project>.pdf`.

**Output:**
- `<project>/problem_graph.json`: Structured question graph.
- `<project>/pages/`: Images of each PDF page.

---

## 2. Engine (`engine.py`)

**Purpose:**  
Reads the problem graph and uses GPT to fill in reasoning-required fields (e.g., image associations, topic tagging, LaTeX conversion), outputting a JSON file for each question.

**How it works:**  
- Loads `<project>/problem_graph.json`.
- For each question, gathers relevant text and images, and constructs a prompt for GPT.
- GPT determines:
  - Where images belong in the text.
  - The topic of each question (using `topics.json` and `subject2id.json`).
  - Whether LaTeX conversion is needed.
  - If a diagram is required.
- Saves the enriched JSON for each question in `<project>/json_folder/`.

**Usage:**
```bash
python engine.py --project <project> --subject <subject>
```
- `<project>`: The project name (must match the folder and PDF).
- `<subject>`: The subject name (must exist in `subject2id.json`).

**Output:**
- `<project>/json_folder/<problem>.json`: Enriched JSON for each question.
- `<project>/images/`: Images referenced in the JSONs.

---

## 3. Postprocessor (`postprocessor.py`)

**Purpose:**  
Converts the per-question JSONs into a single Excel file, uploads images to S3, and updates image references.

**How it works:**  
- Loads all JSONs from `<project>/json_folder/`.
- For each question:
  - Uploads referenced images to S3 and replaces local paths with S3 URLs.
  - Cleans and formats text fields.
  - Flattens the question structure into rows for Excel.
- Outputs an Excel file with all questions and metadata.

**Usage:**
```bash
python postprocessor.py --project <project>
```
- `<project>`: The project name (must match previous steps).

**Output:**
- `<project>v5.xlsx`: Final Excel file with all questions and metadata.

---

## Workflow Summary

1. Place your PDF in `./pdf/<project>.pdf`.
2. Run the preprocessor to generate the problem graph:
   ```
   python preprocessor.py --project <project>
   ```
3. Run the engine to enrich the problem graph using GPT:
   ```
   python engine.py --project <project> --subject <subject>
   ```
4. Run the postprocessor to produce the final Excel file:
   ```
   python postprocessor.py --project <project>
   ```

---

## Environment Setup

1. **Create a `.env` file in the project root with the following content:**
    ```
    OPENAI_API_KEY=your_openai_api_key
    AWS_ACCESS_KEY_ID=your_aws_access_key_id
    AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    AWS_REGION=ap-northeast-2
    S3_BUCKET_NAME=ibguru
    ```
    Replace the values with your actual credentials.

---

## Notes

- All scripts require Python 3 and the dependencies listed above.
- All credentials (OpenAI, AWS) are loaded from the `.env` file using [python-dotenv](https://pypi.org/project/python-dotenv/).
- The pipeline is modular: you can inspect or modify intermediate outputs at each stage.

---

## References

- `topics.json`: Maps subject IDs to topic lists for topic tagging.
- `subject2id.json`: Maps subject names to subject IDs.
- See code comments for further details on each processing step.
