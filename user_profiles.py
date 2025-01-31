from RAG_agent_definition import init_llm

def build_predefined_user_system_prompt(user_profile):
    return f"""You are a {user_profile['gender']},
            living in {user_profile['country']},
            of  {user_profile['age']} years old,
              {user_profile['height']} cm tall,
                {user_profile['weight']} kg heavy.
                to describe you, we can also say that {user_profile['user_information']}.
                ****
                You are suffering the following health conditions:
       {user_profile['health_condition_list']}
        and
        {user_profile['other_health_condition_str']}
        They also have the following allergies and intolerance's: 
        {user_profile['allergies_list']}
        and 
        {user_profile['other_allergies_str']}
        and
        {user_profile['food_intolerance_list']}
        and
        {user_profile['other_food_intolerance_str']}
        ****
        You are using a smart diet plan assistant for:
        {user_profile['reason_to_chat_with_cocineco']}
        ****
                You are not allowed to give answers of more than 5 words.
                You can only answer questions about yourself"""


def init_user_agent_from_profile(user_profile):
    return init_llm(temperature=0), build_predefined_user_system_prompt(user_profile)


predefined_profiles = {
    "Antonio": {
        "user_name": "Antonio",
        "gender": "male",
        "age": 25,
        "height": 180,
        "weight": 90,
        "country": "Spain",
        "other_health_condition_str": "",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "has allergy to gluten",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": """have a sedentary job,
                                    is partying a lot,
                                    does a lot of sport,
                                    doesn't like to cook,""",
        "reason_to_chat_with_cocineco": "I want to lose a bit of weight."
    },
    "Paula": {
        "user_name": "Paula",
        "gender": "female",
        "age": 55,
        "height": 165,
        "weight": 50,
        "country": "Mexico",
        "other_health_condition_str": "",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": "would like to lose a bit of weight.",
        "reason_to_chat_with_cocineco": "I want to lose a bit of weight."
    },
    "Albert": {
        "user_name": "Albert",
        "gender": "male",
        "age": 65,
        "height": 180,
        "weight": 70,
        "country": "France",
        "other_health_condition_str": "",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": "is a retired professional athlete.",
        "reason_to_chat_with_cocineco": "I want to maintain my fitness."
    },
    "Kim": {
        "user_name": "Kim",
        "gender": "male",
        "age": 35,
        "height": 170,
        "weight": 50,
        "country": "India",
        "other_health_condition_str": "Cancer  and is undergoing Chemo has been prescribed blood thinners and multi-vitamins.also suffered a heart attack. ",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": "He consumes alcohol regularly. He updated the meal plan to have Soju on weekends.",
        "reason_to_chat_with_cocineco": "I want to maintain my fitness."
    },
    "Ivo": {
        "user_name": "Ivo",
        "gender": "male",
        "age": 27,
        "height": 187,
        "weight": 95,
        "country": "Lebanon",
        "other_health_condition_str": "",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": "is interested in improving his diet to improve his weight lifting performances.",
        "reason_to_chat_with_cocineco": "I want to improve my weight lifting performances."
    },
    "Ahmad": {
        "user_name": "Ahmad",
        "gender": "male",
        "age": 70,
        "height": 155,
        "weight": 59,
        "country": "United Kingdom",
        "other_health_condition_str": "",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "Celiac disease",
        "user_information": "is a swimmer and feels hungry quite often.",
        "reason_to_chat_with_cocineco": "I want to improve my health."
    },
    "Mia": {
        "user_name": "Mia",
        "gender": "female",
        "age": 27,
        "height": 168,
        "weight": 70,
        "country": "Switzerland",
        "other_health_condition_str": "she is in first trimester of her pregnancy and has low haemoglobin",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": "She enjoys walking. She is vegan with no social habits.She asked for a protein and iron rich meal.",
        "reason_to_chat_with_cocineco": "I want to improve my health."
    },
    "Cherry": {
        "user_name": "Cherry",
        "gender": "female",
        "age": 33,
        "height": 169,
        "weight": 46,
        "country": "Italy",
        "other_health_condition_str": "",
        "health_condition_list": [],
        "allergies_list": [],
        "other_allergies_str": "",
        "food_intolerance_list": [],
        "other_food_intolerance_str": "",
        "user_information": "She wants to improve her weight and strength. She practices Yoga. She has nut allergy.",
        "reason_to_chat_with_cocineco": "I want to  loose weight and gain strength."
    },
}
