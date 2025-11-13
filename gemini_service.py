import requests, json, os

GEMINI_API_KEY = "AIzaSyDNnr_lnZGfQ-9RHZYmfZbdRt3ZQ1hZ9q4"
GEMINI_MODEL = "gemini-2.5-pro"

def list_gemini_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    resp = requests.get(url, timeout=15)
    data = resp.json()
    
    models = []
    if isinstance(data, dict) == True:
        for k, v in data.items():
            if k == 'models' and isinstance(v, list) == True:
                for item in v:
                    if isinstance(item, str) == True:
                        models.append(item)
                    elif isinstance(item, dict) == True:
                        if 'name' in item:
                            models.append(item['name'])
                        elif 'model' in item:
                            models.append(item['model'])
    return models

def get_gemini_order(dataset_name, target_values):
    override_path = os.path.join('data', 'training', dataset_name + '.label_order.json')
    if os.path.exists(override_path) == True:
        try:
            with open(override_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, dict) == True and 'order' in payload and isinstance(payload['order'], list) == True:
                print(f"Using local override for {dataset_name}")
                return payload['order']
        except Exception:
            raise RuntimeError(f"Override file {override_path} is malformed.")

    available_models = list_gemini_models()
    
    model_to_use = None
    if GEMINI_MODEL != None and '2.5' in GEMINI_MODEL and 'pro' in GEMINI_MODEL:
        for m in available_models:
            if GEMINI_MODEL in m:
                model_to_use = m
                break

    if model_to_use == None:
        for m in available_models:
            if 'gemini-2.5' in m and 'pro' in m:
                model_to_use = m
                break

    if model_to_use == None:
        raise RuntimeError(f"No Gemini 2.5 Pro model found.")

    normalized_model = model_to_use.split('/')[-1]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent?key={GEMINI_API_KEY}"
    print(f"Using Gemini model: {normalized_model}")

    prompt = (
        "You are given a JSON array of labels. Return a JSON object with a single key 'order' whose value is an array"
        " containing those labels ordered from least likely to be an exoplanet to most likely. Respond ONLY with JSON."
        "\nLabels: " + json.dumps(target_values)
    )
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, json=data, timeout=30)
    result = response.json()

    texts = []
    def walk(o):
        if isinstance(o, str) == True:
            texts.append(o)
        elif isinstance(o, dict) == True:
            for v in o.values():
                walk(v)
        elif isinstance(o, list) == True:
            for v in o:
                walk(v)
    walk(result)

    for text in texts:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                snippet = text[start:end+1]
                parsed = json.loads(snippet)
                if isinstance(parsed, dict) == True and 'order' in parsed and isinstance(parsed['order'], list) == True:
                    return parsed['order']
        except Exception:
            continue

    raise RuntimeError(f'Invalid Gemini response for {dataset_name}')
