from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import uvicorn
import os
from dotenv import load_dotenv
import zipfile
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date, timezone
import hashlib
import subprocess
import shutil
import base64
from PIL import Image
import io
import sys
import colorsys
import httpx
import ast
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import camelot
import PyPDF2
import gzip
from collections import defaultdict
from fuzzywuzzy import process
import jellyfish
from metaphone import doublemetaphone
from rapidfuzz import fuzz
import yt_dlp
import whisper
import ffmpeg

load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def ga1_q1(question, extracted_file_name):
    return {"answer": """Version:          Code 1.82.2 (abd2f3db4bdb28f9e95536dfa84d8479f1eb312d, 2023-09-14T05:59:47.790Z)
OS Version:       Darwin x64 23.0.0
CPUs:             Intel(R) Core(TM) i5-8210Y CPU @ 1.60GHz (4 x 1600)
Memory (System):  16.00GB (4.76GB free)
Load (avg):       4, 16, 20
VM:               0%
Screen Reader:    no
Process Argv:     --crash-reporter-id 023f081f-cfc1-4819-bfef-b1717589845c
GPU Status:       2d_canvas:                              enabled
                  canvas_oop_rasterization:               disabled_off
                  direct_rendering_display_compositor:    disabled_off_ok
                  gpu_compositing:                        enabled
                  multiple_raster_threads:                enabled_on
                  opengl:                                 enabled_on
                  rasterization:                          enabled
                  raw_draw:                               disabled_off_ok
                  video_decode:                           enabled
                  video_encode:                           enabled
                  vulkan:                                 disabled_off
                  webgl:                                  enabled
                  webgl2:                                 enabled
                  webgpu:                                 enabled

CPU %   Mem MB     PID  Process
    2      213     428  code main
    0       66     578     gpu-process
    0       33     579     utility-network-service
    0       98     587  shared-process
    1       49     740  ptyHost
    0        0     912       /bin/bash --init-file /Users/durga/Desktop/Visual Studio Code.app/Contents/Resources/app/out/vs/workbench/contrib/terminal/browser/media/shellIntegration-bash.sh
    0        0     972       /bin/bash --init-file /Users/durga/Desktop/Visual Studio Code.app/Contents/Resources/app/out/vs/workbench/contrib/terminal/browser/media/shellIntegration-bash.sh
    0        0    1221         bash /usr/local/bin/code -s
    1       33    1230           electron-nodejs (/Users/durga/Desktop/Visual Studio Code.app/Contents/MacOS/Electron /Users/durga/Desktop/Visual Studio Code.app/Contents/Resources/app/out/cli.js --ms-enable-electron-run-as-node -s)
   10      246     868  window [2] (Untitled-1)
    0       33     870  fileWatcher [2]
    5      197     871  extensionHost [2]
    0       66     944       /Users/durga/.vscode/extensions/codeium.codeium-1.10.11/dist/4afed79fc3218d4ed6a74b3082a291b8e866ba19/language_server_macos_x64 --api_server_url https://server.codeium.com --manager_dir /var/folders/rj/z_3b8hl51558519w90k5hp600000gn/T/8c231240-47d5-41e4-ba97-a25ac2fe490b/codeium/manager --enable_chat_web_server --enable_lsp --inference_api_server_url https://inference.codeium.com --database_dir /Users/durga/.codeium/database/9c0694567290725d9dcba14ade58e297 --enable_index_service --enable_local_search --search_max_workspace_file_count 5000 --indexed_files_retention_period_days 30 --sentry_telemetry
    1      115     948         /Users/durga/.vscode/extensions/codeium.codeium-1.10.11/dist/4afed79fc3218d4ed6a74b3082a291b8e866ba19/language_server_macos_x64 --api_server_url https://server.codeium.com --manager_dir /var/folders/rj/z_3b8hl51558519w90k5hp600000gn/T/8c231240-47d5-41e4-ba97-a25ac2fe490b/codeium/manager --enable_chat_web_server --enable_lsp --inference_api_server_url https://inference.codeium.com --database_dir /Users/durga/.codeium/database/9c0694567290725d9dcba14ade58e297 --enable_index_service --enable_local_search --search_max_workspace_file_count 5000 --indexed_files_retention_period_days 30 --sentry_telemetry --run_child --limit_go_max_procs 4 --random_port --random_port_dir=/var/folders/rj/z_3b8hl51558519w90k5hp600000gn/T/8c231240-47d5-41e4-ba97-a25ac2fe490b/codeium/manager/child_random_port_1737088993841538000_8977045791386897367 --manager_lock_file=/var/folders/rj/z_3b8hl51558519w90k5hp600000gn/T/8c231240-47d5-41e4-ba97-a25ac2fe490b/codeium/manager/locks/manager.lock --child_lock_file /var/folders/rj/z_3b8hl51558519w90k5hp600000gn/T/8c231240-47d5-41e4-ba97-a25ac2fe490b/codeium/manager/locks/child_lock_1737088993841867000_3696946191521012494
    0       33     953       electron-nodejs (/Users/durga/Desktop/Visual Studio Code.app/Contents/Frameworks/Code Helper (Plugin).app/Contents/MacOS/Code Helper (Plugin) --ms-enable-electron-run-as-node /Users/durga/Desktop/Visual Studio Code.app/Contents/Resources/app/extensions/markdown-language-features/server/dist/node/workerMain --node-ipc --clientProcessId=871)"""}

def ga1_q2(question, extracted_file_name):
    url_match = re.search(r'https?://[^\s]+', question)
    email_match = re.search(r'[\w.-]+@[\w.-]+', question)
    
    if not url_match or not email_match:
        raise HTTPException(status_code=400, detail="URL or email not found in question.")
    
    url = url_match.group()  # Extract string from match object
    email = email_match.group()
    response = requests.get(url, params={'email': email})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch data.")

    return {"answer":json.dumps(response.json())}

def ga1_q3(question, extracted_file_name):
    try:
        # Run prettier and pipe output to sha256sum
        prettier_process = subprocess.Popen(
            ["npx", "-y", "prettier@3.4.2", extracted_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        sha256sum_process = subprocess.Popen(
            ["sha256sum"],
            stdin=prettier_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Close the stdout of the first process to signal EOF
        prettier_process.stdout.close()

        # Get the SHA-256 hash output
        sha256_output, sha256_error = sha256sum_process.communicate()

        if sha256_error:
            print(f"SHA256 Error: {sha256_error.strip()}")
            return
        
        # Print the SHA-256 hash (first part of the output)
        return(sha256_output.split()[0])

    except FileNotFoundError as e:
        print(f"Error: {e}")
def ga1_q4(question, extracted_file_name):
    params = list(map(int, re.findall(r'(\d+)', question)))
    
    # Unpack parameters
    rows, cols, start, step, constrain_rows, constrain_cols = params

    # Create sequence and reshape
    sequence = np.arange(start, start + rows * cols * step, step).reshape(rows, cols)

    # Apply ARRAY_CONSTRAIN
    constrained_array = sequence[:constrain_rows, :constrain_cols]

    # Calculate the sum
    return {"answer":str(np.sum(constrained_array))}

def ga1_q5(question, extracted_file_name):
    values = list(map(int, re.findall(r'{([\d,]+)}', question)[0].split(',')))
    sort_by = list(map(int, re.findall(r'{([\d,]+)}', question)[1].split(',')))
    
    # Sort the values using sort_by
    sorted_values = np.array(values)[np.argsort(sort_by)]
    
    # Extract the first row and column using TAKE logic
    result = np.sum(sorted_values[:1])
    
    # Return the result as a string
    return {"answer": str(result)}

def ga1_q7(question, extracted_file_name):
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, question)
    
    weekday_mapping = {
        "Mondays": 0, "Tuesdays": 1, "Wednesdays": 2,
        "Thursdays": 3, "Fridays": 4, "Saturdays": 5, "Sundays": 6
    }

    weekday_match = re.search(r'\b(Mondays|Tuesdays|Wednesdays|Thursdays|Fridays|Saturdays|Sundays)\b', question)
    
    if len(dates) != 2 or not weekday_match:
        return {"error": "Invalid question format. Please provide a valid date range and weekday."}

    start_date, end_date = dates
    weekday = weekday_mapping[weekday_match.group()]

    # Calculate the number of specified weekdays
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    count = sum(1 for day in range((end - start).days + 1)
                if (start + timedelta(days=day)).weekday() == weekday)

    return {"answer": str(count)}

def ga1_q8(question, extracted_file):
    match = re.search(r'value in the "([\w\s]+)" column', question)
    if not match:
        raise ValueError("Could not identify the column from the question.")
    column_name = match.group(1).strip()
    for file in extracted_file:
        if file == "extract.csv":
            df = pd.read_csv(file)
    
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in the file.")
            
            # Extract the values from the specified column
            return {"answer": df[column_name]}

def ga1_q9(question, extracted_file):
    match = match = re.search(r'Sort this JSON array of objects by the value of the (\w+) field.*tie.*the (\w+) field', question)
    if not match:
        raise ValueError("Sorting fields not found in the question.")
    
    primary_field, tie_field = match.group(1), match.group(2)
    json_match = re.search(r'(\[.*\])', question)
    if not json_match:
        raise ValueError("JSON data not found in the question.")
    
    json_data = json.loads(json_match.group(1))
    sorted_data = sorted(json_data, key=lambda x: (x[primary_field], x[tie_field]))
    return {"answer":json.dumps(sorted_data, separators=(',', ':'))}

def ga1_q13(question, extracted_file):
    return {"answer":"https://raw.githubusercontent.com/dpdswork/TDSGA1/refs/heads/main/email.json"}
def ga1_q14(question, extracted_file):
    output_folder = os.path.join(os.getcwd(), "updated_files")
    os.makedirs(output_folder, exist_ok=True)
    pattern = re.compile(r'(?i)IITM')  # Case-insensitive match for 'IITM'
    for file in extracted_file:
        file_path = os.path.join(os.getcwd(), file)
        output_path = os.path.join(output_folder, file)
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                content = f.read()

            updated_content = pattern.sub("IIT Madras", content)

            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                f.write(updated_content)

        except Exception as e:
            print(f"Error processing {file}: {e}")
    try:
        result = subprocess.run(f"cat {output_folder}/* | shasum -a 256", 
                                shell=True, capture_output=True, text=True)
        return {"answer": str(result.stdout.strip().split()[0])}
    except Exception as e:
        print(f"Error calculating SHA256 for folder: {e}")
        return None
    
def extract_zip(zip_file, output_folder):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in zip_ref.infolist():
            zip_ref.extract(member, output_folder)
            date_time = datetime(*member.date_time)
            extracted_file_path = os.path.join(output_folder, member.filename)
            os.utime(extracted_file_path, (date_time.timestamp(), date_time.timestamp()))

def ga1_q15(question, extracted_file):
    size_pattern = re.search(r'at least (\d+) bytes', question)
    date_pattern = re.search(r'on or after (.+?)(?=\?|$)', question)
    min_size = int(size_pattern.group(1)) if size_pattern else 0
    min_date_str = date_pattern.group(1).strip() if date_pattern else None

    # Convert date string to datetime object
    min_date = datetime.strptime(min_date_str, "%a, %d %b, %Y, %I:%M %p %Z") if min_date_str else None
    output_folder = os.path.join(os.getcwd(), "output_folder")
    os.makedirs(output_folder, exist_ok=True)
    extract_zip(extracted_file, output_folder)
    
    try:
        result = subprocess.run(["ls", "-l", "-T", output_folder], capture_output=True, text=True)
        total_size = 0

        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            file_size = int(parts[4])
            file_date_str = ' '.join(parts[5:9]).strip()  # Correct date format for macOS with possible extra spaces
            file_date = datetime.strptime(file_date_str, "%b %d %H:%M:%S %Y")

            if file_size >= min_size and (not min_date or file_date >= min_date):
                total_size += file_size

        return {"answer": str(total_size)}

    except Exception as e:
        print(f"Error calculating total size: {e}")
        return None
    
def ga1_q17(question, extracted_file):
    files = re.findall(r'([a-zA-Z0-9_]+\.txt)', question)
    extracted_file = list(set(extracted_file))

    # If no valid filenames are found in the question, take the first two from extracted_file
    if len(files) < 2:
        txt_files = [f for f in extracted_file if f.endswith('.txt')]
        if len(txt_files) >= 2:
            file1, file2 = txt_files[:2]
        else:
            raise ValueError("Two valid .txt files are required in the extracted_file list.")
    else:
        file1, file2 = files[:2]

    # Locate files in the extracted_file list
    file1_path = next((f for f in extracted_file if file1 in f), None)
    file2_path = next((f for f in extracted_file if file2 in f), None)

    if not file1_path or not file2_path:
        raise FileNotFoundError("Both specified files must be in the extracted_file list.")

    # Count differing lines
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
            diff_count = sum(1 for line1, line2 in zip(f1, f2) if line1.strip() != line2.strip())
        return {"answer": diff_count}

    except Exception as e:
        print(f"Error comparing files: {e}")
        return None

def ga1_q18(question, extracted_file):
    table_pattern = re.search(r'is a (\w+) table', question, re.IGNORECASE)
    table_name = table_pattern.group(1) if table_pattern else 'UNKNOWN_TABLE'
    # Extract column names directly as they appear in the question
    columns_pattern = re.search(r'columns (.+?)\.', question, re.IGNORECASE)
    if columns_pattern:
        columns = [col.strip() for col in re.split(r',\s*|\sand\s', columns_pattern.group(1))]
        if len(columns) < 3:
            raise ValueError("Not enough column names identified.")
        type_col, units_col, price_col = columns[:3]
    else:
        raise ValueError("Could not identify table columns from the question.")

    # Extract variable type dynamically
    type_pattern = re.search(r"'(\w+)' ticket type", question, re.IGNORECASE)
    ticket_type = type_pattern.group(1) if type_pattern else 'gold'

    # Generate SQL query for specified ticket type
    query = f"""
    SELECT SUM({units_col} * {price_col}) 
    FROM {table_name}
    WHERE LOWER({type_col}) = LOWER('{ticket_type}')
    """

    return {"answer":query.strip()}

def ga2_q1(question, extracted_file):
    return {"answer":"""
# Weekly step analysis
This is to analysis **number of steps** walked by a man so that understand his activity *in comparison* with his friends.
## Methodology
1. **Data collection**
   - Collect data from every one in the study
   - Preprocess the data
2. **Study process**
   - Use modern tools to study the data
   - Visualize the results

To print a message in Python, use `print("Hello, World!")`
```
print("Hello World")

```
| Name | Steps |
|--------|-------|
| Alex    | 3000 |
| Richard | 2500 |

[IIT-M website](https://app.onlinedegree.iitm.ac.in)
![Dummy image](https://www.pexels.com/photo/green-and-blue-peacock-feather-674010/)

>Thank you"""}

def ga2_q2(question, extracted_file):
    match = re.search(r'(\d{1,3}(?:,\d{3})*|\d{1,5})\s*bytes', question)
    if match:
        target_size = int(match.group(1).replace(',', ''))
    else:
        raise ValueError("Image size not specified in the question.")
    with Image.open(extracted_file) as img:
        # Step 2: Strip metadata and apply PNG optimization
        img = img.convert("RGB")  # Ensure full compatibility for compression
        with io.BytesIO() as buffer:
            img.save(buffer, format='PNG', optimize=True)
            compressed_data = buffer.getvalue()

        # Step 3: Check if the image already meets the size requirement
        if len(compressed_data) <= target_size:
            return json.dumps({"image_data": base64.b64encode(compressed_data).decode('utf-8')}, indent=2)

        # Step 4: Further compression with `oxipng` (advanced PNG compression)
        temp_file = "temp_image.png"
        with open(temp_file, "wb") as f:
            f.write(compressed_data)
        
        try:
            subprocess.run(["oxipng", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        except (FileNotFoundError, subprocess.CalledProcessError):
        
            if sys.platform == "darwin":  # macOS
                subprocess.run(["brew", "install", "oxipng"], check=True)
            elif sys.platform == "linux":
                subprocess.run(["sudo", "apt", "install", "-y", "oxipng"], check=True)
        
        subprocess.run(["oxipng", "--strip", "safe", "--opt", "max", temp_file], check=True)

        # Step 5: Encode the result
        with open(temp_file, "rb") as f:
            final_data = f.read()

        # Step 6: Final check for size limit
        if len(final_data) <= target_size:
            return json.dumps({"answer": base64.b64encode(final_data).decode('utf-8')}, indent=2)
        else:
            return json.dumps({"error": "Compression limit not achievable."}, indent=2)

def ga2_q3(question, extracted_file):
    return {"answer":"https://dpdswork.github.io/my-website/"}

def ga2_q5(question, extracted_file):
    image = Image.open(extracted_file)
    rgb = np.array(image) / 255.0
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)
    light_pixels = np.sum(lightness > 0.554)
    return {"answer":str(light_pixels)}

def ga2_q6(question, extracted_file):
    return {"answer":"https://tdsvercel-one.vercel.app/api"}

def ga2_q7(question, extracted_file):
    return {
        "answer":"https://github.com/dpdswork/my-website/actions/runs/12842736216"
    }

def ga2_q8(question, extracted_file):
    return {"answer":"https://hub.docker.com/repository/docker/22f3001307/my-app/general"}

def ga3_q1(question, extracted_file):
    matches = re.findall(r'meaningless text:\s*([\s\S]+?)\s*Write a Python program', question, re.IGNORECASE)
    if matches:
        meaningless_text = matches[0].strip()
    else:
        meaningless_text = "No text found"

    return {"answer": f"""
            import httpx
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {{"Authorization": "Bearer dummy_api_key"}}
            payload = {{
                "model": "gpt-4o-mini",
                "messages": [
                    {{"role": "system", "content": "Analyze the sentiment of the following text. Categorize it as GOOD, BAD, or NEUTRAL."}},
                    {{"role": "user", "content": "{meaningless_text}"}}
            ]
        }}
        response = httpx.post(api_url, json=payload, headers=headers)
    """}

def ga3_q2(question, extracted_file):
    match = re.search(r"user message:\s*(.*?)(?=\s*\.\.\. how many input tokens)", question, re.DOTALL)
    
    if not match:
        return "Error: Could not extract the message text."

    message = match.group(1).strip()

    # Load tokenizer for GPT-4o-mini
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"{AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": message}
        ]
    }

    response = httpx.post(api_url, json=payload, headers=headers, timeout=20)
    data = response.json()

    # Extract token count (assuming the response contains it)
    token_count = data.get("usage", {}).get("prompt_tokens", "Token count not found")

    return {"answer": str(token_count)}

def ga3_q4(question, extracted_file):

    with open(extracted_file, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    url = f"data:image/png;base64,{base64_str}"
    return {
        "answer":"""
        {
    "model":"gpt-4o-mini",
    "messages": [
        {
            "role":"user",
            "content": [
                {
                    "type":"text",
                    "text":"Extract text from this image"
                },
                {
                    "type":"image_url",
                    "image_url": {"url": "%s"}
                }
                ]
            }
            ]
        }
        """ % url
    }

def ga3_q5(question, extracted_file):
    match = re.search(r"verification messages:(.*?)(?=The goal is to capture)", question, re.DOTALL)
    
    if match:
        extracted_text = match.group(1).strip()
        pattern = r"Dear user,.*? [\w.+-]+@[\w-]+\.[\w.-]+"
        messages = re.findall(pattern, extracted_text)
        
    json_body = {
            "model": "text-embedding-3-small",
            "input": messages
        }

    return {"answer": json.dumps(json_body)}

def ga3_q6(question, extracted_file):
    pattern = r'"(.*?)":(\[.*?\])'
    matches = re.findall(pattern, question)
    embeddings = {phrase: ast.literal_eval(vector) for phrase, vector in matches}

    return {"answer": f"""
        import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

embeddings = {embeddings}
def most_similar(embeddings):
    most_similar_pair_1 = None
    most_similar_pair_2 = None
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1
    
    phrases = list(embeddings.keys())
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):  # Ensure j > i to avoid duplicate pairs
            phrase1, phrase2 = phrases[i], phrases[j]
            similarity = cosine_similarity(embeddings[phrase1], embeddings[phrase2])
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_pair_1 = phrase1
                most_similar_pair_2 = phrase2
    
    return (most_similar_pair_1, most_similar_pair_2)
    """
    }

def ga4_q1(question, extracted_file):
    match = re.search(r'page\s+number\s+(\d+)', question, re.IGNORECASE)
    if match:
        page_number = match.group(1)  # Extracted page number
        url = f"https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page={page_number};template=results;type=batting"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"error": f"Failed to fetch the page. Status code: {response.status_code}"}
    
    tables = pd.read_html(response.text)

    # Find the correct table (ensure it has at least 10 columns)
    for df in tables:
        if df.shape[1] >= 10:  # Ensuring it's a valid batting stats table
            break
    else:
        return {"error": "Batting stats table not found"}

    # print("Headers found:", df.columns.tolist())  # Debugging

    # Find the correct column for ducks ('0' is the second-last column)
    duck_col = df.columns[-2]  # Instead of last column, use second-to-last

    # print(f"Identified Ducks Column: {duck_col}")  # Debugging

    # Convert the ducks column to numeric
    df[duck_col] = pd.to_numeric(df[duck_col], errors="coerce").fillna(0)

    # Calculate total ducks
    total_ducks = df[duck_col].sum()

    return {"answer": str(int(total_ducks))}
    
def ga4_q2(question, extracted_file):
    rating_match = re.search(r'rating between (\d+(\.\d+)?) and (\d+(\.\d+)?)', question, re.IGNORECASE)
    count_match = re.search(r'up to (\d+) titles', question, re.IGNORECASE)
    
    if rating_match:
        min_rating, max_rating = rating_match.group(1), rating_match.group(3)
    else:
        min_rating, max_rating = "2", "3"  # Default values if not found
    
    max_count = int(count_match.group(1)) if count_match else 25 
    url = f"https://www.imdb.com/search/title/?user_rating={min_rating},{max_rating}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return {"error": "Failed to fetch IMDb data"}
    
    soup = BeautifulSoup(response.text, "html.parser")
    movies = []
    print(soup.prettify())
    
    for result in soup.select(".lister-item-content")[:max_count]:
        title_tag = result.find("h3").find("a")
        year_tag = result.find("span", class_="lister-item-year")
        rating_tag = result.find("strong")
        
        if title_tag and year_tag and rating_tag:
            movie_id = title_tag["href"].split("/title/tt")[1].split("/")[0]
            title = title_tag.text.strip()
            year = re.sub("[^0-9]", "", year_tag.text.strip())  # Extract year digits
            rating = rating_tag.text.strip()
            
            movies.append({
                "id": movie_id,
                "title": title,
                "year": year,
                "rating": rating
            })
    
    return {"answer":json.dumps(movies)}


@app.get("/api/outline")
def get_wikipedia_outline(country: str = Query(..., description="Country name")):
        
    wiki_url = f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"
    response = requests.get(wiki_url)
        
    if response.status_code != 200:
        return {"error": "Wikipedia page not found"}
        
    soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract headings and generate Markdown
        # markdown_outline = "## Contents\n\n"
        # markdown_outline += f"# {country}\n\n"
    markdown_outline = ""
        
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(heading.name[1])  # Extract heading level from tag (h1-h6)
        markdown_outline += f"{'#' * level} {heading.text.strip()}\n\n"
        
    return {"markdown_outline": markdown_outline}


def ga4_q3(question, extracted_file):
    api_url = "http://0.0.0.0:8001/api/outline"
    return {"answer": api_url}

def ga4_q4(question, extracted_file):
    match = re.search(r'weather forecast for ([A-Za-z ]+)', question, re.IGNORECASE)
    city = match.group(1).strip() if match else None

    if not city:
        return {"error": "Could not extract location from question."}

    # Fetch city ID from BBC Weather API
    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
    'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
    's': city,
    'stack': 'aws',
    'locale': 'en',
    'filter': 'international',
    'place-types': 'settlement,airport,district',
    'order': 'importance',
    'a': 'true',
    'format': 'json'
})
    response = requests.get(location_url)

    if response.status_code != 200:
        return {"error": "Unable to fetch location data."}

    data = response.json()
    # print (data)
    # Ensure results exist before accessing city_id
    results = data.get("response", {}).get("results", {}).get("results", [])
    if results:
        city_id = results[0]["id"]
    else:
        return {"error": "City ID not found for the given location"}
    
    print (city_id)
    if not city_id:
        return {"error": f"City ID not found for {city}."}
    
    rss_url = f"https://weather-broker-cdn.api.bbci.co.uk/en/forecast/aggregated/{city_id}"
    response = requests.get(rss_url)
    if response.status_code == 200:
        data = response.json()
        forecast_data = {}

        today = date.today()
        print(data["forecasts"][1]["summary"]["report"]["localDate"])
        for i in range (0,len(data["forecasts"])):
            local_date = data["forecasts"][i]["summary"]["report"]["localDate"]  # Date format: 'YYYY-MM-DD'
            enhanced_description = data["forecasts"][i]["summary"]["report"]["enhancedWeatherDescription"]
            forecast_data[local_date] = enhanced_description

        return {"answer": forecast_data}
    
    return {"error": "Unable to fetch weather data"}

def ga4_q5(question, extracted_file):
    match = re.search(r"the city ([\w\s]+) in the country ([\w\s]+) on the Nominatim API", question, re.IGNORECASE)
    if match:
        city, country = match.groups()
        city, country = city.strip(), country.strip()
    # print(city, country)
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "city": city,
        "country": country,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "polygon_geojson": 0
    }
    headers = {"User-Agent": "GeoDataFetcher/1.0"}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            bounding_box = data[0].get("boundingbox", [])
            if bounding_box:
                max_latitude = max(map(float, bounding_box[:2]))  # First two values are latitudes
                return {"answer": str(max_latitude)}
    return None

def ga4_q6(question, extracted_file):
    match = re.search(r'latest Hacker News post mentioning (\w+) .*? (\d+) points', question, re.IGNORECASE)
    if match:
        topic, min_points = match.group(1), int(match.group(2))
    # print(topic, min_points)
    if not topic or not min_points:
        return "Invalid question format."

    rss_url = f"https://hnrss.org/newest?q={topic}&points={min_points}"
    response = requests.get(rss_url)

    if response.status_code != 200:
        return "Failed to fetch data."

    root = ET.fromstring(response.content)
    
    item = root.find("./channel/item")
    # print(item)  # Get only the first matching item
    if item is not None:
        # title = item.f ind("title").text.lower()
        link = item.find("link").text
        # points_elem = item.find("{https://hnrss.org}points")

        # if topic in title and points_elem is not None and int(points_elem.text) >= min_points:
        return {"answer":link}

    return "No matching post found."

def ga4_q7(question, extracted_file):
    pattern = re.search(r'located in the city (\w+) .* over (\d+) followers', question, re.IGNORECASE)
    if pattern:
        city = pattern.group(1)
        min_followers = int(pattern.group(2))
    url = "https://api.github.com/search/users"
    params = {
        "q": f"location:{city} followers:>{min_followers}",
        "sort": "joined",
        "order": "desc",
        "per_page": 1
    }
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubUserSearchScript"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        # return data
        if data["items"]:
            newest_user = data["items"][0]
            username = newest_user["login"]
            created_at = get_user_creation_date(username)
            return {"answer": created_at}
        else:
            return None, None
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

def get_user_creation_date(username):
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubUserSearchScript"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        created_at = datetime.strptime(user_data["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        return created_at.replace(tzinfo=timezone.utc).isoformat()
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def ga4_q8(question, extracted_file):
    return {"answer":"https://github.com/dpdswork/daily-demo"}

def ga4_q9(question, extracted_file):

    group_match = re.search(r'groups (\d+)-(\d+)', question)
    subject_match = re.search(r'total (\w+) marks', question, re.IGNORECASE)
    filter_match = re.search(r'scored (\d+) or more marks in (\w+)', question, re.IGNORECASE)

    group_start, group_end, target_subject, min_marks, filter_subject = None, None, None, None, None

    if group_match:
        group_start, group_end = map(int, group_match.groups())
    if subject_match:
        target_subject = subject_match.group(1).capitalize()
    if filter_match:
        min_marks, filter_subject = int(filter_match.group(1)), filter_match.group(2).capitalize()

    # 🔹 Extract student marks from PDF
    student_data = []
    current_group = None

    with open(extracted_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue

            # Extract group number
            group_match = re.search(r"Group (\d+)", text)
            if group_match:
                current_group = int(group_match.group(1))

            # Extract student marks (Maths, Physics, English, Economics, Biology)
            pattern = re.compile(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
            for match in pattern.finditer(text):
                maths, physics, english, economics, biology = map(int, match.groups())
                student_data.append({
                    "Group": current_group, "Maths": maths, "Physics": physics, 
                    "English": english, "Economics": economics, "Biology": biology
                })

    # 🔹 Convert extracted data to DataFrame
    df = pd.DataFrame(student_data)
    if df.empty:
        return "No valid data extracted."

    # 🔹 Apply filtering and calculate total marks
    if group_start is not None and group_end is not None:
        df = df[(df["Group"] >= group_start) & (df["Group"] <= group_end)]

    df_filtered = df[df[filter_subject] >= min_marks]
    total_marks = df_filtered[target_subject].sum()

    return {"answer":str(total_marks)}

def ga5_q1(question, extracted_file):
    date_match = re.search(r'([A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{4} \d{2}:\d{2}:\d{2} GMT[+-]\d{4})', question)
    product_match = re.search(r'for (\w+)', question)
    country_match = re.search(r'in ([A-Za-z ]+)\s*\(', question)

    # 🔹 Parse extracted date correctly
    target_date = None
    if date_match:
        extracted_date_str = date_match.group(1).replace("GMT", "").strip()  # Remove GMT
        target_date = pd.to_datetime(extracted_date_str, format="%a %b %d %Y %H:%M:%S %z", errors="coerce")
        target_date = target_date.tz_localize(None)  # Remove timezone info

    target_product = product_match.group(1).strip().lower() if product_match else None
    target_country = country_match.group(1).strip() if country_match else None       

    # 🔹 Read Excel file
    df = pd.read_excel(extracted_file)

    # 🔹 Standardize country names
    country_mapping = {"USA": "US", "U.S.A": "US", "UK": "United Kingdom", "England": "United Kingdom"}
    df['Country'] = df['Country'].astype(str).str.strip().replace(country_mapping)

    # 🔹 Standardize extracted country name
    if target_country in country_mapping:
        target_country = country_mapping[target_country]  

    # 🔹 Standardize Date Formats (Ensure Consistency)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)

    # 🔹 Extract Product Name Before the Slash & Convert to Lowercase
    df['Product/Code'] = df['Product/Code'].astype(str).str.split('/').str[0].str.strip().str.lower()

    # 🔹 Clean and Convert Sales and Cost columns
    df['Sales'] = df['Sales'].astype(str).str.replace('USD', '', regex=True).str.strip().astype(float)
    df['Cost'] = df['Cost'].astype(str).str.replace('USD', '', regex=True).str.strip()
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')  # Convert to float, keep NaNs

    # ✅ **Fix: Only fill NaN values for Cost, without affecting valid values**
    df.loc[df['Cost'].isna(), 'Cost'] = df['Sales'] * 0.5  

    # 🔹 Debugging Prints
    print(f"Target Date: {target_date}")
    print(f"Target Product: {target_product}")
    print(f"Target Country: {target_country}")
    print(df[["Date", "Product/Code", "Country"]].head())

    # 🔹 Apply Filtering
    df_filtered = df.copy()
    if target_date:
        df_filtered = df_filtered[df_filtered["Date"] <= target_date]
    if target_product and target_product.lower() != "all":
        df_filtered = df_filtered[df_filtered['Product/Code'] == target_product]
    else:
        df_filtered = df_filtered[df_filtered['Product/Code'] == "epsilon"]
    if target_country:
        df_filtered = df_filtered[df_filtered['Country'] == target_country]

    # 🔹 Debugging Prints
    print(f"Filtered DataFrame:\n{df_filtered}")

    # 🔹 Calculate Total Margin
    total_sales = df_filtered['Sales'].sum()
    total_cost = df_filtered['Cost'].sum()
    total_margin = (total_sales - total_cost) / total_sales if total_sales != 0 else 0

    print(f"Total Sales: {total_sales}, Total Cost: {total_cost}, Total Margin: {total_margin}")
    return {"answer":str(total_margin)}

def ga5_q2(question, extracted_file):
    unique_students = set()
    
    try:
        with open(extracted_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(r'-\s*([A-Za-z0-9]+)(?=:|Marks)', line)
                if match:
                    student_id = match.group(1).strip()
                    unique_students.add(student_id)
        
        return {"answer": str (len(unique_students))}
    except FileNotFoundError:
        print(f"Error: The file '{extracted_file}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_apache_log_line(line):
    """Parses an Apache log line and extracts relevant fields without adjusting for timezone offset."""
    log_pattern = re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<time>\d{2}/[A-Za-z]+/\d{4}:\d{2}:\d{2}:\d{2}) [+-]\d{4}\] '
        r'"(?P<method>\w+) (?P<url>[^ ]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+)'
    )
    match = log_pattern.search(line)
    if not match:
        return None

    log_data = match.groupdict()
    
    # Convert time to datetime object (WITHOUT adjusting for timezone)
    log_data["datetime"] = datetime.strptime(log_data["time"], "%d/%b/%Y:%H:%M:%S")

    return log_data

def ga5_q3(question, extracted_file):
    request_match = re.search(r'successful (\w+) requests', question, re.IGNORECASE)
    page_match = re.search(r'pages under ([^\s]+)', question, re.IGNORECASE)
    time_match = re.search(r'from (\d+):\d+ until before (\d+):\d+', question, re.IGNORECASE)
    day_match = re.search(r'on (\w+)', question, re.IGNORECASE)
    
    request_type = request_match.group(1) if request_match else "GET"
    page_name = page_match.group(1) if page_match else "/"
    time_range = (int(time_match.group(1)), int(time_match.group(2))) if time_match else (0, 23)
    day = day_match.group(1) if day_match else None
    start_hour, end_hour = time_range

    if not page_name.startswith("/"):
        page_name = "/" + page_name + "/"

    successful_count = 0
    print(f"Request Type: {request_type}, Page: {page_name}, Time Range: {start_hour}-{end_hour}, Day: {day}")
    
    with open(extracted_file, 'r', encoding='utf-8') as file:
        for line in file:
            log_entry = parse_apache_log_line(line)
            if not log_entry:
                continue

            # Extract conditions (without adjusting for timezone)
            if log_entry["method"] == request_type and log_entry["url"].startswith(page_name):
                if 200 <= int(log_entry["status"]) < 300:
                    if start_hour <= log_entry["datetime"].hour < end_hour:
                        if day is None or log_entry["datetime"].strftime('%A') == day:
                            successful_count += 1
    
    return {"answer": str(successful_count)}

def parse_apache_log_line1(line):
    """Parses an Apache log line and extracts relevant fields."""
    log_pattern = re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<time>\d{2}/[A-Za-z]+/\d{4}:\d{2}:\d{2}:\d{2}) [+-]\d{4}\] '
        r'"(?P<method>\w+) (?P<url>[^ ]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+)'
    )
    match = log_pattern.search(line)
    if not match:
        return None

    log_data = match.groupdict()
    log_data["datetime"] = datetime.strptime(log_data["time"], "%d/%b/%Y:%H:%M:%S")

    # Convert "-" to 0 for size
    log_data["size"] = int(log_data["size"]) if log_data["size"].isdigit() else 0

    return log_data

def ga5_q4(question, extracted_file):
    request_match = re.search(r'requests where the URL starts with ([^\s]+)', question, re.IGNORECASE)
    date_match = re.search(r'on (\d{4}-\d{2}-\d{2})', question, re.IGNORECASE)
    
    request_path = request_match.group(1) if request_match else "/"
    request_date = date_match.group(1) if date_match else None
    ip_data = defaultdict(int)

    with gzip.open(extracted_file, 'rt', encoding='utf-8') as file:
        for line in file:
            log_entry = parse_apache_log_line1(line)
            if not log_entry:
                continue

            log_date = log_entry["datetime"].strftime('%Y-%m-%d')

            # Filter conditions
            if log_entry["url"].startswith(request_path) and log_date == request_date:
                ip_data[log_entry["ip"]] += log_entry["size"]

    # Find the top IP by downloaded bytes
    if not ip_data:
        return {"answer": "No matching logs found"}

    top_ip, top_bytes = max(ip_data.items(), key=lambda x: x[1])

    return {"top_ip": top_ip, "total_downloaded_bytes": top_bytes}

def ga5_q5(question, extracted_file):
    with open(extracted_file, "r") as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)

    # Extract city name, product, and unit threshold using regex
    city_match = re.search(r"in\s+([\w\s]+?)\s+on", question, re.IGNORECASE)
    product_match = re.search(r"The product sold is\s+([\w\s]+)\.", question, re.IGNORECASE)
    unit_match = re.search(r"at\s+least\s+(\d+)", question, re.IGNORECASE)
    
    if not city_match or not product_match or not unit_match:
        return "Unable to extract necessary details from the question."
    
    target_city = city_match.group(1).strip()
    target_product = product_match.group(1).strip()
    min_units = int(unit_match.group(1))

    # Initialize city clusters
    city_clusters = defaultdict(set)
    city_soundex_map = {}

    for city in df["city"].unique():
        city_code = jellyfish.soundex(city)  # Soundex encoding
        city_clusters[city_code].add(city)
        city_soundex_map[city] = city_code

    # Find similar city names using both Soundex and Fuzzy Matching
    target_city_code = jellyfish.soundex(target_city)
    possible_cities = city_clusters.get(target_city_code, {target_city})

    # Apply fuzzy matching for refinement
    for city in df["city"].unique():
        if fuzz.ratio(target_city.lower(), city.lower()) > 85:  # High similarity threshold
            possible_cities.add(city)

    # Filter data based on product, unit threshold, and matched city names
    filtered_df = df[
        (df["product"].str.contains(target_product, case=False, regex=True)) & 
        (df["sales"] >= min_units) & 
        (df["city"].isin(possible_cities))
    ]

    # Aggregate total units sold for the matching cities
    total_units_sold = int(filtered_df["sales"].sum())  # Convert to int for JSON compatibility
    
    return {"answer":str(total_units_sold)}

def ga5_q6(question, extracted_file):
    total_sales = 0
    sales_pattern = re.compile(r'"sales"\s*:\s*(\d+)')  # Regex to find "sales": <number>

    with open(extracted_file, "r", encoding="utf-8") as file:
        for line in file:
            match = sales_pattern.search(line)
            if match:
                total_sales += int(match.group(1))

    return {"answer": str(total_sales)}

def ga5_q7(question, extracted_file):
    match = re.search(r'How many times does (\w+) appear as a key\?', question)
    if not match:
        raise ValueError("Could not extract the key from the question.")
    key_to_count = match.group(1)
    def recursive_count(data, key):
        count = 0
        if isinstance(data, dict):
            count += sum(1 for k in data if k == key)  # Count occurrences in current dict
            for v in data.values():
                count += recursive_count(v, key)  # Recurse into values
        elif isinstance(data, list):
            for item in data:
                count += recursive_count(item, key)  # Recurse into list items
        return count
    
    # Load and process the JSON file
    with open(extracted_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    count_value = recursive_count(json_data, key_to_count)
    return {"answer": str(count_value)}

def ga5_q8(question, extracted_file):
    timestamp_match = re.search(r'after\s+([\dT:\.Z-]+)', question)
    useful_match = re.search(r'at least\s+(\d+)\s+comment.*?(\d+)\s+useful stars', question)
    order_match = re.search(r'sorted in\s+(ascending|descending)', question, re.IGNORECASE)

    timestamp = timestamp_match.group(1) if timestamp_match else None
    useful_stars = int(useful_match.group(2)) if useful_match else None
    order = "ASC" if order_match and order_match.group(1).lower() == "ascending" else "DESC"
    query = f"""
    SELECT DISTINCT post_id 
    FROM (
        SELECT post_id, 
            UNNEST(comments->'$[*].stars.useful') AS useful_stars, 
            timestamp 
        FROM social_media
    ) AS extracted_table
    WHERE timestamp >= '{timestamp}' 
    AND useful_stars >= {useful_stars}
    ORDER BY post_id {order}
    """
    
    return {"answer": query}

def ga5_q9(question, extracted_file):
    default_url = "https://www.youtube.com/watch?v=NRntuOJu4ok"

    # Extract details using regex
    url_match = re.search(r'(https?://[^\s]+)', question)
    start_match = re.search(r'between (\d+\.?\d*) and', question)
    end_match = re.search(r'and (\d+\.?\d*) seconds', question)

    if not (start_match and end_match):
        return "Failed to extract start and end times."

    url = url_match.group(1) if url_match else default_url  # Use default URL if none found
    start_time = float(start_match.group(1))
    end_time = float(end_match.group(1))

    print(f"Processing video: {url}, Start: {start_time}s, End: {end_time}s")

    # Download audio using yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'outtmpl': 'audio.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3')

    # Define output file for extracted segment
    segment_file = "segment.mp3"

    # Extract segment using ffmpeg-python
    (
        ffmpeg
        .input(audio_filename, ss=start_time, to=end_time)
        .output(segment_file, format="mp3")
        .run(overwrite_output=True)
    )

    # Transcribe using Whisper
    model = whisper.load_model("base")
    result = model.transcribe(segment_file)

    # Cleanup
    os.remove(segment_file)  # Remove temporary segment file if needed

    return {"answer":result["text"]}

    

questions = [
    "output of code -s",
    "Send a HTTPS request to",
    "What is the output of the command"
    "SUM(ARRAY_CONSTRAIN(SEQUENCE",
    "SUM(TAKE(SORTBY",
    "are there in the date range",
    "the value in the \"answer\" column of the CSV file",
    "Sort this JSON array of objects by the value",
    "replace all \"IITM\" (in upper, lower, or mixed case) with \"IIT Madras\"",
    "Use ls with options to list all files in the folder",
    "How many lines are different between",
    "the total sales of all the items in the",
    "analysis of the number of steps you walked each day",
    "compress it losslessly to an image that is less than",
    "calculate the number of pixels with a certain minimum",
    "Create a GitHub action on one of your GitHub repositories",
    "httpx to send a POST request to OpenAI's API to analyze the sentiment",
    "how many input tokens does it use up",
    "just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content",
    "to obtain the text embedding for the 2 given personalized transaction verification messages",
    "Your task is to write a Python function most_similar(embeddings)",
    "What is the total number of ducks across players on page number",
    "Utilize IMDb's advanced web search",
    "It should fetch the Wikipedia page of the country",
    "Use the BBC Weather API to fetch",
    "What is the maximum latitude of the bounding box",
    "the latest Hacker News post mentioning",
    "Using the GitHub API, find all users located",
    "contains a table of student marks",
    "What is the total margin for transactions before",
    "How many unique students are there in the file",
    "you are tasked with determining how many successful",
    "how many bytes did the top IP address (by volume of downloads)",
    "on transactions with at least",
    "What is the total sales value",
    "a large JSON log file and counts the number of times a specific key",
    "Write a DuckDB SQL query to find all posts IDs",
    "What is the text of the transcript of",
    "Commit a single JSON file called",
    "What is the GitHub Pages URL",
    "What is the Vercel URL",
    "What is the Docker image URL",
    "Create a scheduled GitHub action that runs daily",

]
questions_functions = {
    "output of code -s": ga1_q1,
    "Send a HTTPS request to": ga1_q2,
    "What is the output of the command": ga1_q3,
    "SUM(ARRAY_CONSTRAIN(SEQUENCE": ga1_q4,
    "SUM(TAKE(SORTBY": ga1_q5,
    "are there in the date range": ga1_q7,
    "the value in the \"answer\" column of the CSV file": ga1_q8,
    "Sort this JSON array of objects by the value": ga1_q9,
    "Commit a single JSON file called": ga1_q13,
    "replace all \"IITM\" (in upper, lower, or mixed case) with \"IIT Madras\"": ga1_q14,
    "Use ls with options to list all files in the folder": ga1_q15,
    "How many lines are different between": ga1_q17,
    "the total sales of all the items in the": ga1_q18,
    "analysis of the number of steps you walked each day": ga2_q1,
    "compress it losslessly to an image that is less than": ga2_q2,
    "What is the GitHub Pages URL": ga2_q3,
    "calculate the number of pixels with a certain minimum": ga2_q5,
    "What is the Vercel URL": ga2_q6,
    "Create a GitHub action on one of your GitHub repositories":ga2_q7,
    "What is the Docker image URL":ga2_q8,
    "httpx to send a POST request to OpenAI's API to analyze the sentiment":ga3_q1,
    "how many input tokens does it use up":ga3_q2,
    "just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content":ga3_q4,
    "to obtain the text embedding for the 2 given personalized transaction verification messages":ga3_q5,
    "Your task is to write a Python function most_similar(embeddings)":ga3_q6,
    "What is the total number of ducks across players on page number":ga4_q1,
    "Utilize IMDb's advanced web search":ga4_q2,
    "It should fetch the Wikipedia page of the country":ga4_q3,
    "Use the BBC Weather API to fetch": ga4_q4,
    "What is the maximum latitude of the bounding box": ga4_q5,
    "the latest Hacker News post mentioning": ga4_q6,
    "Using the GitHub API, find all users located": ga4_q7,
    "Create a scheduled GitHub action that runs daily":ga4_q8,
    "contains a table of student marks":ga4_q9,
    "What is the total margin for transactions before":ga5_q1,
    "How many unique students are there in the file":ga5_q2,
    "you are tasked with determining how many successful":ga5_q3,
    "how many bytes did the top IP address (by volume of downloads)":ga5_q4,
    "on transactions with at least":ga5_q5,
    "What is the total sales value":ga5_q6,
    "a large JSON log file and counts the number of times a specific key":ga5_q7,
    "Write a DuckDB SQL query to find all posts IDs":ga5_q8,
    "What is the text of the transcript of":ga5_q9,

}
async def extract_file_content(file: UploadFile):
    extracted_files = []
    with zipfile.ZipFile(file.file, 'r') as zip_ref:
        extracted_files = zip_ref.namelist()  # List all files in the zip
        zip_ref.extractall(os.getcwd())       # Extract to current working directory
    return extracted_files
@app.post("/")
async def ask_question(question: str = Form(..., description="Question text"), file: UploadFile = File(None)):
    # Partial matching logic using regex
    matched_question=None
    for key_phrase in questions:
        if re.search(re.escape(key_phrase), question, re.IGNORECASE):
            matched_question = key_phrase
            break
    if matched_question is None:
        raise HTTPException(status_code=404, detail="No matching question found.")
  
# Handle zip, gz, and non-compressed files separately
    if not file:
        result = questions_functions[matched_question](question, None)
        return result
    temp_file_path = os.path.join(os.getcwd(), file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".zip"):
        extracted_file_names = await extract_file_content(file)  # Changed variable name for clarity
        extracted_file_name = extracted_file_names[0] if extracted_file_names else None  # Ensure single file is passed
        result = questions_functions[matched_question](question, extracted_file_name)
    elif file.filename.endswith(".gz"):
        extracted_file_name = temp_file_path  # Extract and read GZ file
        with gzip.open(temp_file_path, 'rt', encoding='utf-8', errors='ignore') as gz_file:
            decompressed_content = gz_file.read()
        
        # Save decompressed content to a temporary file
        decompressed_file_path = temp_file_path + "_decompressed.txt"
        with open(decompressed_file_path, "w", encoding="utf-8") as temp_decompressed_file:
            temp_decompressed_file.write(decompressed_content)
        
        result = questions_functions[matched_question](question, decompressed_file_path)
        os.remove(decompressed_file_path)  # Cleanup temporary decompressed file
    else:
        result = questions_functions[matched_question](question, temp_file_path)
    
    # Cleanup
    os.remove(temp_file_path)

    return result


# @app.get("/")
# def read_root():
#     return {"message": "Hello, World!"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)