# streamlit_app.py (v5 - Added Azure OpenAI Recommendations)
# UI for Python Script Energy Consumption Prediction + Recommendations

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ast
import os
import requests
import zipfile
import io
import tempfile
import shutil
import re
from openai import AzureOpenAI # Import Azure OpenAI client

# --- Configuration ---
MODEL_FILENAME = 'random_forest_energy_model.joblib'
FEATURES_ORDER = ['LOC', 'No_of_Functions', 'No_of_Classes', 'No_of_Loops',
                  'Loop_Nesting_Depth', 'No_of_Conditional_Blocks', 'Import_Score',
                  'I/O Calls']

# --- Feature Extraction Code ---

library_weights = {
    "torch": 10, "tensorflow": 10, "jax": 9, "keras": 9, "transformers": 9, "lightgbm": 8,
    "xgboost": 8, "catboost": 8, "sklearn": 7, "scikit-learn": 7, "pandas": 6, "numpy": 6,
    "dask": 7, "polars": 6, "matplotlib": 4, "seaborn": 4, "plotly": 5, "bokeh": 5,
    "altair": 4, "cv2": 6, "PIL": 4, "imageio": 3, "scikit-image": 4, "nltk": 5, "spacy": 6,
    "gensim": 5, "requests": 2, "httpx": 2, "urllib": 1, "aiohttp": 3, "fastapi": 4, "flask": 3,
    "django": 5, "openpyxl": 3, "csv": 1, "json": 1, "sqlite3": 2, "sqlalchemy": 3, "h5py": 4,
    "pickle": 2, "os": 1, "sys": 1, "shutil": 1, "glob": 1, "pathlib": 1, "math": 1,
    "statistics": 1, "scipy": 5, "datetime": 1, "time": 1, "calendar": 1, "re": 1,
    "argparse": 1, "typing": 1, "logging": 1, "threading": 2, "multiprocessing": 3,
    "concurrent": 3, "subprocess": 3, "random": 1, "uuid": 1, "hashlib": 1, "base64": 1,
    "decimal": 1, "boto3": 6, "google.cloud": 6, "azure": 6, "pyspark": 9, "IPython": 2,
    "jupyter": 2
}
file_io_funcs = {'open', 'read', 'write', 'remove', 'rename', 'copy', 'seek', 'tell', 'flush'}

class FeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.max_loop_depth = 0; self.current_depth = 0; self.file_io_calls = 0
    def visit_For(self, node):
        self.current_depth += 1; self.max_loop_depth = max(self.max_loop_depth, self.current_depth)
        self.generic_visit(node); self.current_depth -= 1
    def visit_While(self, node):
        self.current_depth += 1; self.max_loop_depth = max(self.max_loop_depth, self.current_depth)
        self.generic_visit(node); self.current_depth -= 1
    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name): func_name = node.func.id
        elif isinstance(node.func, ast.Attribute): func_name = node.func.attr
        if func_name in file_io_funcs: self.file_io_calls += 1
        self.generic_visit(node)


# Modified to also return the source code string
def extract_features_and_code_from_file(file_path):
    """
    Reads a Python file, extracts static code features and source code.
    Returns (features_dict, source_code_string) or (None, None) on failure.
    """
    source_code = None
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()
    except Exception as e:
        print(f"Error reading script file {file_path}: {e}")
        st.error(f"Could not read file: {os.path.basename(file_path)}")
        return None, None # Return None for both

    try:
        tree = ast.parse(source_code)
    except Exception as e:
         print(f"Error parsing script {file_path} with AST: {e}")
         st.error(f"Could not parse file (invalid Python?): {os.path.basename(file_path)}")
         return None, source_code # Return code even if parsing fails, maybe useful

    # Calculate features...
    num_lines = len(source_code.splitlines())
    num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    num_classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    num_loops = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    num_conditional_blocks = sum(isinstance(node, ast.If) for node in ast.walk(tree))
    num_imports = 0; imported_libs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                lib = name.name.split('.')[0];
                if lib: imported_libs.add(lib)
                num_imports += 1
        elif isinstance(node, ast.ImportFrom):
             if node.module: lib = node.module.split('.')[0];
             if lib: imported_libs.add(lib)
             num_imports += 1
    weighted_import_score = sum(library_weights.get(lib, 2) for lib in imported_libs)
    extractor = FeatureExtractor(); extractor.visit(tree)
    features_dict = {
        'LOC': num_lines, 'No_of_Functions': num_functions, 'No_of_Classes': num_classes,
        'No_of_Loops': num_loops, 'Loop_Nesting_Depth': extractor.max_loop_depth,
        'No_of_Conditional_Blocks': num_conditional_blocks, 'Import_Score': weighted_import_score,
        'I/O Calls': extractor.file_io_calls
    }
    return features_dict, source_code # Return both

def predict_for_features(model, features_dict):
    # (Same as before)
    try:
        input_features = {key: [pd.to_numeric(value, errors='coerce')] for key, value in features_dict.items()}
        input_df = pd.DataFrame(input_features, columns=FEATURES_ORDER)
        if input_df.isnull().values.any():
             st.error("Error: Non-numeric values found in extracted features during prediction step.")
             return None
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction step: {e}")
        return None

# --- Azure OpenAI Function ---
# @st.cache_data # Optionally cache OpenAI responses for a short time
def get_openai_recommendations(source_code, features_dict):
    """Sends code and features to Azure OpenAI for recommendations."""
    recommendations = "Could not retrieve recommendations." # Default message
    try:
        # --- IMPORTANT: Configure Credentials ---
        # Uses Streamlit Secrets Management (secrets.toml) - Recommended for deployment
        # Ensure you have created .streamlit/secrets.toml with your Azure keys
        # Check if secrets are loaded
        if not all(k in st.secrets for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"]):
             st.error("Azure OpenAI credentials missing in Streamlit Secrets (secrets.toml).")
             st.info("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME are set in .streamlit/secrets.toml")
             return "Credentials configuration missing."

        api_key = st.secrets["AZURE_OPENAI_API_KEY"]
        azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
        deployment_name = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

        # Check if values are actually set (secrets might exist but be empty)
        if not all([api_key, azure_endpoint, api_version, deployment_name]):
            st.error("One or more Azure OpenAI credentials in Streamlit Secrets are empty.")
            return "Credentials configuration incomplete."


        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )

        # --- Construct the Prompt ---
        features_str = "\n".join([f"- {key.replace('_', ' ')}: {value}" for key, value in features_dict.items()])

        prompt_messages = [
            {"role": "system", "content": "You are an expert Python programmer specialized in code optimization for energy efficiency. Analyze the provided code and its static features to suggest specific, actionable changes that would likely reduce its energy consumption during execution."},
            {"role": "user", "content": f"""Please analyze the following Python code for potential energy optimizations.

        Consider factors like:
        - Algorithmic efficiency (e.g., unnecessary computations, better data structures)
        - Loop optimizations (e.g., reducing iterations, vectorization)
        - I/O operations (e.g., batching, buffering, efficient file handling)
        - Library usage (e.g., choosing lighter alternatives if possible, efficient use of heavy libraries)
        - Concurrency/Parallelism (potential benefits or overhead)
        - Memory usage patterns

        Provide specific, actionable recommendations on how to modify the code to reduce energy consumption. Focus on practical changes and explain the reasoning. Structure recommendations clearly, perhaps using bullet points.

        Code Features:
        {features_str}

        Source Code:
        ```python
        {source_code}
        Recommendations:

        Also give the percentage improvement in the Energy Efficiency after doing the recommended changes in the following format:
        Recommendation | Percentage Improvement
        Algorithmic Efficiency | 5-10%
        Loop Optimizations | 1-3%
        """}
        ]

        # --- Make API Call ---
        st.write("_Contacting Azure OpenAI... (This may take a moment)_") # Give user feedback
        response = client.chat.completions.create(
            model=deployment_name, # Your deployment name
            messages=prompt_messages,
            temperature=0.5, # Lower temperature for more focused recommendations
            max_tokens=4000, # Increased slightly for potentially longer recommendations
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        # --- Extract Response ---
        if response.choices:
            recommendations = response.choices[0].message.content.strip()
            if not recommendations: # Handle empty response case
                recommendations = "AI model returned an empty recommendation."
        else:
            recommendations = "No recommendations received from API (response structure unexpected)."

    except ImportError:
        st.error("The 'openai' library is not installed. Please add it to requirements.txt and reinstall.")
        recommendations = "OpenAI library not found."
    except KeyError as e:
        # This catches cases where a secret isn't defined in secrets.toml
        st.error(f"Azure OpenAI credential '{e}' not found in Streamlit Secrets (secrets.toml).")
        recommendations = f"Credential configuration missing: {e}"
    except Exception as e:
        st.error(f"Error calling Azure OpenAI API: {e}")
        recommendations = f"Error fetching recommendations: {e}"

    return recommendations
#--- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🐍 Sustainable Code Reviewer (Prototype)")

@st.cache_resource # Cache the loaded model
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file '{filename}' not found. Ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load model '{filename}'. Error: {e}")
        st.stop()

#--- Load Model ---
#This function is defined above, using @st.cache_resource
loaded_model = load_model(MODEL_FILENAME)
st.success(f"Prediction Model loaded successfully.")

#--- User Input ---
st.header("Input")
input_path_or_url = st.text_input(
"Enter public GitHub repository URL:",
placeholder="https://github.com/skills/introduction-to-github"
)

analyze_button = st.button("Analyze and Predict")

#--- Analysis and Prediction Output Area ---

files_to_process = []
scan_source_description = ""
temp_dir_context = None
extracted_repo_root = None
base_path_for_relative = None
target_subdir_in_repo = None

# --- Determine Input Type and Get Files ---

if analyze_button:
    # Check if input is a GitHub URL
    if input_path_or_url.startswith(('http://', 'https://')) and 'github.com' in input_path_or_url:
            scan_source_description = f"GitHub source: {input_path_or_url}"
            st.info(f"Processing {scan_source_description}")

            # --- Improved URL Parsing ---
            match = re.match(r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(/(.*))?)?", input_path_or_url)
            if not match:
                st.error("Could not parse GitHub URL structure. Please provide URL to repo root or subdirectory.")
                st.stop()
            user, repo, branch, _, subdir = match.groups()
            repo = repo.replace(".git", "")
            target_subdir_in_repo = subdir.strip('/') if subdir else None
            repo_url_base = f"https://github.com/{user}/{repo}"
            st.write(f"Detected Repository Base: {repo_url_base}")
            if target_subdir_in_repo: st.write(f"Detected Target Subdirectory: {target_subdir_in_repo}")
            # --- End Improved URL Parsing ---

            # Attempt download using common branches (use specified branch first if available)
            potential_branches = [branch] if branch else ['main', 'master']
            repo_zip_content = None
            for b in potential_branches:
                potential_zip_url = f"{repo_url_base}/archive/refs/heads/{b}.zip"
                st.write(f"Attempting to download zip from branch: {b}...")
                try:
                    response = requests.get(potential_zip_url, stream=True, timeout=30)
                    response.raise_for_status()
                    repo_zip_content = response.content
                    st.write(f"Successfully downloaded zip for branch: {b}")
                    break
                except requests.exceptions.RequestException as e:
                     st.write(f"Could not download zip for branch '{b}': {e}")
                except Exception as e:
                     st.write(f"An unexpected error occurred downloading branch '{b}': {e}")

            if not repo_zip_content:
                st.error("Could not download repository zip. Check URL or repo structure (e.g., branch name).")
                st.stop()

            # Extract to temporary directory
            try:
                temp_dir_context = tempfile.TemporaryDirectory()
                temp_dir = temp_dir_context.name
                st.write(f"Extracting repository to temporary location...")
                with zipfile.ZipFile(io.BytesIO(repo_zip_content)) as zf:
                    zf.extractall(temp_dir)
                    zip_root_folder = zf.namelist()[0].split('/')[0] # e.g., 'repo-main'
                    extracted_repo_root = os.path.join(temp_dir, zip_root_folder)
                st.write("Extraction complete.")

                # Determine the starting path for os.walk
                scan_start_path = extracted_repo_root
                base_path_for_relative = extracted_repo_root # For display
                if target_subdir_in_repo:
                    potential_subdir_path = os.path.join(extracted_repo_root, target_subdir_in_repo.replace('%20', ' '))
                    if os.path.isdir(potential_subdir_path):
                         scan_start_path = potential_subdir_path
                         base_path_for_relative = scan_start_path # Make path relative to subdir
                         st.write(f"Scanning specifically within subdirectory: {target_subdir_in_repo}")
                    else:
                         st.warning(f"Subdirectory '{target_subdir_in_repo}' not found in extracted repo. Scanning entire repository.")

                # Find Python files starting from the scan_start_path
                for root, dirs, files in os.walk(scan_start_path):
                    dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', '__pycache__', '.git']]
                    for file in files:
                        if file.endswith(".py"): files_to_process.append(os.path.join(root, file))

            except Exception as e:
                st.error(f"Error during zip extraction or file scanning: {e}")
                if temp_dir_context: temp_dir_context.cleanup()

    elif os.path.exists(input_path_or_url):
        with st.spinner("Accessing source and finding Python files..."):
            base_path_for_relative = input_path_or_url # Store base path
            if os.path.isfile(input_path_or_url) and input_path_or_url.endswith(".py"):
                scan_source_description = f"local file: {input_path_or_url}"
                st.info(f"Processing {scan_source_description}")
                files_to_process.append(input_path_or_url)
                base_path_for_relative = os.path.dirname(input_path_or_url) # Use dir for relative path
            elif os.path.isdir(input_path_or_url):
                scan_source_description = f"local directory: {input_path_or_url}"
                st.info(f"Processing {scan_source_description}")
                for root, dirs, files in os.walk(input_path_or_url):
                    dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', '__pycache__', '.git']]
                    for file in files:
                        if file.endswith(".py"): files_to_process.append(os.path.join(root, file))
            else:
                st.error(f"Local path is not a directory or a .py file: {input_path_or_url}")
    else:
        st.error(f"Input path or URL not found or not recognized: {input_path_or_url}")
            
        # --- End Determine Input Type ---

# --- Process Files ---
st.header("Results")
results_placeholder = st.container() # Use a container to group results
if not files_to_process:
    st.warning("No Python files found to process.")
else:
    overall_success_count = 0

    with results_placeholder:
        for file_path in files_to_process:
            display_path = os.path.basename(file_path) # Default
            try: # Try getting relative path
                 if base_path_for_relative and os.path.commonpath([base_path_for_relative, file_path]) == os.path.normpath(base_path_for_relative):
                      display_path = os.path.relpath(file_path, base_path_for_relative)
            except ValueError: display_path = os.path.basename(file_path)

            st.subheader(f"Results for: {display_path}")

            # Extract features AND source code
            features_dict, source_code = extract_features_and_code_from_file(file_path) # Handles its own errors via st.error

            if features_dict and source_code:
                # Display features
                st.write("🔍 **Extracted Features:**")
                output_str = "\n".join([f"  • {key.replace('_', ' '):<25} : {value}" for key, value in features_dict.items()])
                st.code(output_str, language=None)

                # Predict Energy
                prediction = predict_for_features(loaded_model, features_dict) # Handles its own errors via st.error

                if prediction is not None:
                    st.success(f"**Predicted Energy: {prediction:.2f} joules**")
                    overall_success_count += 1

                    # Get OpenAI Recommendations
                    st.write("💡 **Fetching Energy Saving Recommendations...**")
                    # Using a spinner specific to the API call
                    with st.spinner("Contacting Azure OpenAI..."):
                         recommendations = get_openai_recommendations(source_code, features_dict)
                    st.markdown("**Recommendations:**")
                    st.markdown(recommendations) # Display recommendations using markdown
                # No 'else' needed as predict_for_features shows st.error

            # No 'else' needed as extract_features_and_code_from_file shows st.error

            st.divider() # Add divider between files

# Cleanup temporary directory
if temp_dir_context:
    try:
        temp_dir_context.cleanup()
        st.write("Temporary directory cleaned up.")
    except Exception as e:
        st.warning(f"Could not automatically clean up temp directory. Error: {e}")

