# LightRag Persona implementation
## Technologies used
- Python
- Tiktoken
- Lightrag
- Motor
- FastApi
- Uvicorn

## Installation Guide
- Install Python 3.10 or older
- On the terminal run `git clone https://github.com/AlexMcBex/lightRag-Persona.git`
- `cd lightRag-Persona`
- `python3 -m venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `touch .env` to create env file
-  insert your .env variables in .env

## Starting guide
- on the terminal run `uvicorn app:app --reload --port 9000`, change 9000 with whatever other port if desired
- access the api with a `POST` request at the endpoint `/match-user` including a "persona" and a "user_id" in the req.body
