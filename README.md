
![CocinEco-](https://github.com/user-attachments/assets/3e3c7846-ea11-45e3-ae9f-94575fc36ae8)


# CocinEco

This repository contains the code developed by [A3I-Data Science](https://a3i-datascience.github.io/) at the occasion of a dataton proposed by [Datais](https://www.datais.es/dataton-sostenibilidad) in Madrid in November 2024.


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

### Installing the environment
- Clone the repository
- Run `uv sync --all-groups` to automatically create a virtual environment (using the appropriate python version) and install all the dependencies as specified in `uv.lock`

### Managing dependencies
- To add/remove dependencies to the project, use `uv add` or `uv remove`. uv will then automatically ensure that versions are compatible. Examples provided below:

```
uv add streamlit
uv remove pandas
uv add "pandas>=2.0.0"
```

### Committing changes
- This project uses pre-commit hooks. There are two ways to use it:
    - [Install pre-commit system-wide](https://pre-commit.com/#install)
    - Use the version of pre-commit that is already included as a dependency of this project

- In both cases, `pre-commit install` should be run once, to install the pre-commit hooks.
- If using the bundled version, it is necessary to activate the environment before running the above command:

``` source .venv/bin/activate```

- If committing a change that adds/updates dependencies, it is required by streamlit cloud to build and commit the `requirements.txt` file:

``` uv pip compile pyproject.toml -o requirement.txt ```

### Running the project locally
- Create an Open AI Key the key [here](https://platform.openai.com/api-keys), and add it to a `.env` file (use `.env.template` as a template)
```
cp .env.template .env
```

- Run the following command:
```
uv run streamlit run Main_Menu.py
```
- Alternatively, you can first activate the environment, and then simply run streamlit:
```
source .venv/bin/activate
streamlit run Main_Menu.py
```


## FAQ

### Why do we need pysqlite-binary?
- Due to an incompatible version of sqlite available by default on the streamlit cloud platform, it is necessary to provide the sqlite binary via python's package management. It is not really needed for local development, but ensuring that we are using the same version of sqlite both locally and on the cloud can't harm us!
