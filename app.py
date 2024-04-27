from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import base64
import requests

app = Flask(__name__)
cors = CORS(app, origins=["http://127.0.0.1:5500", "http://127.0.0.1:5000"])
# Set up OpenAI API key
api_key = "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"


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Get the uploaded image
    image_file = request.files.get('image')
    if image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        base64_image = request.form.get('image')

    # Check the language parameter
    language = request.form.get('language', 'english')

    # Send the image to OpenAI Vision API for disease detection
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Upon receiving the image, look at it and give me the exact name of the disease affecting the plant in relation to diseases found in kenya. The disease should be specific and do not give different answers when the same image is uploaded again, conduct a thorough analysis on the image to give me the exact name of the disease. Bold the disease name. Do not include the name kenya.In addition to the name give me the symptoms of the disease in one brief paragraph with the heading symptoms bolded this should begin in a new pararaph. If the image is not a crop tell the user that we only diagnose crops only the image doesnt look like a crop"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    disease_name_english = f"{response.json()['choices'][0]['message']['content']}"

    client = openai.Client(api_key=api_key)

    treatment_prompt = f"Give me a brief but detailed accurate ways and methods of getting rid of the '{disease_name_english}' disease affecting the crop, and also provide the exact medicine found in kenya and even ferilizers to be used, explaining how to apply it and how to protect the crops from future occurance of the disease. The response should be in a way of advising a farmer in kenya. Bold Itilize the disease name, also arrange your results in an appealing format numbering and bodling the ways before expalining them, also starting with heading the the xplanation bellow it. If there is not disease detected just tell the farmer that you crop is dong well and give ways to keep the crop in it health status Also do not include the name kenya. "
    treatment_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": treatment_prompt}
        ]
    )
    treatment_info_english = treatment_response.choices[0].message.content

    if language == 'kiswahili':
        # Translate the disease name to Kiswahili
        disease_name_translation_prompt = f"Translate this exact text to Kiswahili and do not change anything just translate the way it is: {disease_name_english}"
        disease_name_translation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": disease_name_translation_prompt}
            ]
        )
        disease_name_kiswahili = disease_name_translation_response.choices[0].message.content

        # Translate the existing English treatment information to Kiswahili
        translation_prompt = f"Translate this text to Kiswahili and do not change anything just translate the way it is: {treatment_info_english}"
        translation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": translation_prompt}
            ]
        )
        treatment_info_kiswahili = translation_response.choices[0].message.content

        # Create a response object with the image, disease name, and treatment information
        response_data = {
            'image': base64_image,
            'disease_name': disease_name_kiswahili,
            'treatment_info': treatment_info_kiswahili
        }
    else:
        # Create a response object with the image, disease name, and treatment information
        response_data = {
            'image': base64_image,
            'disease_name': disease_name_english,
            'treatment_info': treatment_info_english
        }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)