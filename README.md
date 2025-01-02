# Chatbot_financial_statement

## Major Update 3
- Having universal code 
- Ready to deploy
- Setup 2 Docker image, MongoDB and Text2SQL
- If you have gpu, you can also use Text Embedding Inference (TEI) image for fast embedding
- Gonna make an image in the future 

## Prompting Strategy
- General: 2-step Text2sql. First asking LLM to analyze the problem and choose which category do they want to access. Then adding snapshot of the table into prompt, so it can correctly select the right column.
- Reasoning: After having snapshot, ask LLM to generate SQL directly to solve the problem
- Partial sql. Instead of query to find the solution, breakdown steps and solve it one-by-one
- Include debugging

## Setup guide (Currently bug)

- Make `run.sh` file executable
```bash
chmod +x run.sh
```

- For CPU (Using OpenAI Embedding)
```bash
./run.sh --force True --openai True
```

- For GPU (Self-hosted Embedding Server)
```bash
./run.sh local-embedding --force True --local True
```
- For GPU, Including Reranker 
```bash
./run.sh local-model --force True --local True 
```

## Setup guide temp

- Run docker image
```bash
docker-compose up -d
```
or with GPU
```bash
docker-compose --profile local-embedding --profile local-reranker up -d
```

- Manually create `test_db` database 

- Setup conda env
```bash
conda create -n text2sql
conda activate text2sql
pip install -r requirements.txt
```

- Install llm lib
```bash
git clone https://github.com/hung20gg/llm.git
```

- Setup database + embedding
```bash
python3 setup.py --force True --openai True
```
or with GPU
```bash
python3 setup.py --force True --local True
```

- Run streamlit
```bash
streamlit run home.py
```

## DB In the pipeline
- ChromaDB (Storing the embedding)
- PostgreSQL (Storing the data)
- MongoDB (Storing the user message)




Check and add the index for full-text search in [ETL\index_full_text_search.md](ETL\index_full_text_search.md)
