from RAG_agent_definition import init_llm


def build_predefined_user_system_prompt(user_profile):
    return f"""You are a {user_profile['gender']}, 
            living in {user_profile['country']},
            of  {user_profile['age']} years old, 
              {user_profile['size']} cm tall,
                {user_profile['weight']} kg heavy.
                to describe you, we can also say that {user_profile['user_information']}.
                You are not allowed to give answers of more than 5 words. 
                You can only answer questions about yourself"""


def init_user_agent_from_profile(user_profile):
    return init_llm(temperature=0), build_predefined_user_system_prompt(user_profile)


predefined_profiles = {
    "Antonio": {
        "user_name": "Antonio",
        "gender": "male",
        "age": 25,
        "size": 180,
        "weight": 90,
        "country": "Spain",
        "user_information": """have a sedentary job, 
                                    has allergy to gluten,
                                    is partying a lot,
                                    does a lot of sport,
                                    doesn't like to cook,
                                    wants to lose a bit of weight.""",
    },
    "Paula": {
        "user_name": "Paula",
        "gender": "female",
        "age": 55,
        "size": 165,
        "weight": 50,
        "country": "Mexico",
        "user_information": "would like to lose a bit of weight.",
    },
    "Albert": {
        "user_name": "Albert",
        "gender": "male",
        "age": 65,
        "size": 180,
        "weight": 70,
        "country": "France",
        "user_information": "is a retired professional athlete and is looking to maintain his fitness.",
    },
    "Kim": {
        "user_name": "Kim",
        "gender": "male",
        "age": 35,
        "size": 170,
        "weight": 50,
        "country": "India",
        "user_information": "has Cancer and is undergoing Chemo. He also suffered a heart attack.He has been prescribed blood thinners and multi-vitamins. He consumes alcohol regularly. He updated the meal plan to have Soju on weekends.",
    },
    "Ivo": {
        "user_name": "Ivo",
        "gender": "male",
        "age": 27,
        "size": 187,
        "weight": 95,
        "country": "Lebanon",
        "user_information": "is interested in improving his diet to improve his weight lifting performances.",
    },
    "Ahmad": {
        "user_name": "Ahmad",
        "gender": "male",
        "age": 70,
        "size": 155,
        "weight": 59,
        "country": "United Kingdom",
        "user_information": "He has Celiac disease and is a swimmer and feels hungry quite often.",
    },
    "Mia": {
        "user_name": "Mia",
        "gender": "female",
        "age": 27,
        "size": 168,
        "weight": 70,
        "country": "Switzerland",
        "user_information": "She is in the first trimester of her pregnancy and has low haemoglobin. She enjoys walking. She is vegan with no social habits.She asked for a protein and iron rich meal.",
    },
    "Cherry": {
        "user_name": "Cherry",
        "gender": "female",
        "age": 33,
        "size": 169,
        "weight": 46,
        "country": "Italy",
        "user_information": "She wants to improve her weight and strength. She practices Yoga. She has nut allergy.",
    },
}
