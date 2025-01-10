
![CocinEco-](https://github.com/user-attachments/assets/3e3c7846-ea11-45e3-ae9f-94575fc36ae8)


# CocinEco

This repository contains the code developed by [A3I-Data Science](https://a3i-datascience.github.io/) at the occasion of a dataton proposed by [Datais](https://www.datais.es/dataton-sostenibilidad) in Madrid in November 2024.


## Acknowledgment

* This flask has been developed using this template https://github.com/arora-r/chatapp-with-voice-and-openai-outline.

## Project Scope

Optimization of nutritional diets with Generative AI

The objective of this challenge is to develop a model and an application using generative AI to optimize nutritional diets, guaranteeing a balanced and adequate intake for different populations. Participants will need to address the design of solutions that not only promote health, but are also customizable and accessible, taking into account the preferences and nutritional needs of users.

Projects will be evaluated based on the effectiveness of the model in creating balanced diets, innovation in the use of generative AI, the usability of the developed application and the clarity of the presentation of results. It is recommended to use the following data sets: Apparent Intake (based on household consumption and expenditure surveys) and Suite of Food Security Indicators 2.

## Data sets:

### FAO Data:

* https://www.fao.org/faostat/en/#data/HCES
* https://www.fao.org/faostat/en/#data/FS
* https://www.fao.org/faostat/en/#data/QCL
* https://www.fao.org/faostat/en/#data

### Nutrition Guidelines References:

* https://iris.who.int/bitstream/handle/10665/326261/9789241515856-eng.pdf
* https://applications.emro.who.int/docs/EMROPUB_2019_en_23536.pdf?ua=1
* https://files.magicapp.org/guideline/a3fe934f-6516-460d-902f-e1c7bbcec034/published_guideline_7330-1_1.pdf
* https://www.who.int/news-room/fact-sheets/detail/healthy-diet
* https://www.who.int/activities/developing-nutrition-guidelines

### Recipes:
* https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m

## Technical references

* https://python.langchain.com/docs/introduction/
* https://huggingface.co/learn/cookbook/en/rag_evaluation
* https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval?utm_source=profile&utm_medium=reader2
* https://github.com/langchain-ai/rag-from-scratch
* https://www.coursera.org/learn/building-gen-ai-powered-applications/

## Quickstart

### Prerequisites
- Install [uv](https://github.com/astral-sh/uv)
- Install [pre-commit](https://pre-commit.com/) (Optional)


### Environment Setup
1. Clone the repo
2. Run the following to create an virtual environment for the project and install all required dependencies.

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
pre-commit install
```

3. Create an Open AI Key the key [here](https://platform.openai.com/api-keys), and add it to a `.env` file (use `.env.template` as a template)
```
cp .env.template .env
```
**Warning** there multiple types of key that you can create on open AI platform

### Running the project

4. Start the server by running the following command
```
streamlit run app.py
```
