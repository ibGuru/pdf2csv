# PDF2CSV Official SAQ

This project processes exam problem sets and mark schemes in PDF format, extracting structured question data and converting it into Excel/CSV format. The workflow is designed for high-fidelity digitization of complex question sets, including diagrams and mark schemes, using OpenAI Vision and S3 for image hosting.

## Overview

- **script.py**: Extracts the structure and content of each question from a problem set PDF and its corresponding mark scheme PDF, using OpenAI Vision for context inference. Generates per-question JSON files with all relevant data, including diagrams and mark schemes.
- **postprocessing.py**: Converts the per-question JSON files into a single Excel file, uploading all referenced images to S3 and replacing local paths with S3 URLs.

## Environment Variables

Create a `.env` file in the project root with the following keys:

```
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-northeast-2
```

## Input Files

- Place your problem set PDF and its corresponding mark scheme PDF in the `pdf/` directory.
- The mark scheme PDF must be named as `{problem_set_name}_markscheme.pdf` (e.g., `Biology_paper_2__TZ1_HL.pdf` and `Biology_paper_2__TZ1_HL_markscheme.pdf`).
- Ensure the names match exactly except for the `_markscheme` suffix.
- Problem source in excel column is fully dependent on input file names. 
- Recommend to name pdf like (Subject_name)_(Problem_source_Format + "-*" + .pdf) form. * character is replaced with problem number automatically. (e.g Chemistry_N20-TZ0-P1-*(HL).pdf)

## Usage

### 1. Extract Questions and Mark Schemes

Run `script.py` to process the PDFs and generate per-question JSON files.

```bash
python script.py --project_name <problem_set_name> --subject_id <subject_id>
```

- `<problem_set_name>`: The base name of your PDF files (without extension or `_markscheme`), e.g., `Biology_paper_2__TZ1_HL`
- `<subject_id>`: The subject ID as required by your topics.json (e.g., `6`)

Example:
```bash
python script.py --project_name Biology_paper_2__TZ1_HL --subject_id 6
```

### 2. Convert to Excel and Upload Images

Run `postprocessing.py` to convert the JSON files to Excel and upload images to S3.

```bash
python postprocessing.py --project_name <problem_set_name>
```

Example:
```bash
python postprocessing.py --project_name Biology_paper_2__TZ1_HL
```

## Output

- Per-question JSON files: `./<project_name>/json_folder/`
- Excel file: `<project_name>v7.xlsx` in the project root
- All images referenced in the Excel will be uploaded to S3 and replaced with S3 URLs.

## Notes

- All processing is based on OpenAI Vision and may introduce probabilistic errors due to the complexity of visual inference.
- The entire process is argument-driven; do not hardcode variables in the scripts.
- Ensure your `.env` file is correctly set up before running the scripts.

## Directory Structure

```
pdf2csv_official_saq/
├── pdf/
│   ├── Biology_paper_2__TZ1_HL.pdf
│   ├── Biology_paper_2__TZ1_HL_markscheme.pdf
├── script.py
├── postprocessing.py
├── README.md
├── .env
└── ...
```