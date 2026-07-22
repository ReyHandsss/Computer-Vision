import os
import re
import time
import base64
import requests
import pandas as pd
from jiwer import cer
from tqdm import tqdm
from datetime import datetime

# KONFIGURASI

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-vl-7b"

IMAGE_FOLDER = "images/test"
LABEL_FOLDER = "labels/test"

OUTPUT_CSV = "hasil_ocr_plat_nomor.csv"
SUMMARY_FILE = "summary.txt"

PROMPT = """
Read the Indonesian license plate in this image.

Return only the license plate text.

Example:
B1234XYZ
"""

# LOAD CLASS

with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# CEK LM STUDIO

def check_lmstudio():

    try:
        response = requests.get(
            "http://127.0.0.1:1234/v1/models",
            timeout=5
        )

        response.raise_for_status()

        print("[INFO] LM Studio Connected\n")

    except requests.exceptions.RequestException:

        print("=" * 60)
        print("LM Studio belum aktif.")
        print("Silakan jalankan LM Studio terlebih dahulu.")
        print("=" * 60)
        exit()


# BACA GROUND TRUTH

def read_ground_truth(label_path):

    chars = []

    with open(label_path, "r") as f:

        for line in f:

            data = line.strip().split()

            class_id = int(data[0])

            x_center = float(data[1])

            chars.append((x_center, classes[class_id]))

    chars.sort(key=lambda x: x[0])

    return "".join(c for _, c in chars)


# ENCODE IMAGE

def encode_image(image_path):

    with open(image_path, "rb") as f:

        return base64.b64encode(f.read()).decode("utf-8")


# OCR DENGAN LM STUDIO

def predict_plate(image_path):

    image_base64 = encode_image(image_path)

    payload = {

        "model": MODEL_NAME,

        "messages": [

            {

                "role": "user",

                "content": [

                    {

                        "type": "text",

                        "text": PROMPT

                    },

                    {

                        "type": "image_url",

                        "image_url": {

                            "url": f"data:image/jpeg;base64,{image_base64}"

                        }

                    }

                ]

            }

        ],

        "temperature": 0,

        "max_tokens": 20

    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:

        print("Request Error:", e)

        return ""

    result = response.json()

    prediction = result["choices"][0]["message"]["content"]

    prediction = prediction.upper()

    prediction = prediction.replace(" ", "")
    prediction = prediction.replace("-", "")
    prediction = prediction.replace("\n", "")
    prediction = prediction.replace("PLAT", "")
    prediction = prediction.replace("LICENSE", "")
    prediction = prediction.replace("NUMBER", "")
    prediction = prediction.replace(":", "")

    prediction = re.sub(r"[^A-Z0-9]", "", prediction)

    return prediction


# MAIN

def main():

    check_lmstudio()

    print("=" * 60)
    print("INDONESIAN LICENSE PLATE OCR")
    print("=" * 60)
    print("Model      :", MODEL_NAME)
    print("Date       :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Image Path :", IMAGE_FOLDER)
    print("Label Path :", LABEL_FOLDER)
    print("=" * 60)
    print("Prompt:")
    print(PROMPT.strip())
    print("=" * 60)

    results = []

    image_files = sorted([

        f for f in os.listdir(IMAGE_FOLDER)

        if f.lower().endswith((".jpg", ".jpeg", ".png"))

    ])

    print(f"Total Images : {len(image_files)}")
    print()

    start_time = time.time()

    for image_name in tqdm(image_files, desc="Processing Images"):

        image_path = os.path.join(IMAGE_FOLDER, image_name)

        label_name = os.path.splitext(image_name)[0] + ".txt"

        label_path = os.path.join(LABEL_FOLDER, label_name)

        if not os.path.exists(label_path):

            print(f"[WARNING] Label tidak ditemukan : {label_name}")

            continue

        ground_truth = read_ground_truth(label_path)

        try:

            prediction = predict_plate(image_path)

        except Exception as e:

            print(f"[ERROR] {image_name} : {e}")

            prediction = ""

        cer_score = cer(ground_truth, prediction)

        accuracy = round((1 - cer_score) * 100, 2)

        results.append({

            "image": image_name,

            "ground_truth": ground_truth,

            "prediction": prediction,

            "CER_score": round(cer_score, 4),

            "Accuracy(%)": accuracy

        })

        time.sleep(0.2)

    # SAVE CSV

    df = pd.DataFrame(results)

    df.to_csv(

        OUTPUT_CSV,

        index=False,

        encoding="utf-8-sig"

    )

    elapsed = time.time() - start_time

    avg_cer = df["CER_score"].mean()

    char_accuracy = (1 - avg_cer) * 100

    perfect = (df["CER_score"] == 0).sum()

    failed = len(df) - perfect

    success_rate = perfect / len(df) * 100

    print()
    print("=" * 60)
    print("OCR Evaluation Summary")
    print("=" * 60)
    print(f"CSV File            : {OUTPUT_CSV}")
    print(f"Total Images        : {len(df)}")
    print(f"Perfect OCR         : {perfect}")
    print(f"Incorrect OCR       : {failed}")
    print(f"Success Rate        : {success_rate:.2f}%")
    print(f"Average CER         : {avg_cer:.4f}")
    print(f"Character Accuracy  : {char_accuracy:.2f}%")
    print(f"Execution Time      : {elapsed:.2f} seconds")
    print("=" * 60)

    wrong = df[df["CER_score"] > 0]

    if len(wrong) > 0:

        print("\nContoh OCR yang belum tepat:")
        print(wrong[["image",
                     "ground_truth",
                     "prediction",
                     "CER_score"]].head(10))

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:

        f.write("INDONESIAN LICENSE PLATE OCR\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date               : {datetime.now()}\n")
        f.write(f"Model              : {MODEL_NAME}\n")
        f.write(f"Total Images       : {len(df)}\n")
        f.write(f"Perfect OCR        : {perfect}\n")
        f.write(f"Incorrect OCR      : {failed}\n")
        f.write(f"Success Rate       : {success_rate:.2f}%\n")
        f.write(f"Average CER        : {avg_cer:.4f}\n")
        f.write(f"Character Accuracy : {char_accuracy:.2f}%\n")
        f.write(f"Execution Time     : {elapsed:.2f} seconds\n")

    print("\nSample Results")
    print(df.head(10))

    print(f"\nSummary disimpan pada : {SUMMARY_FILE}")


# RUN PROGRAM

if __name__ == "__main__":

    main()