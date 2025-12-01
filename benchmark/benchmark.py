# benchmark_runner.py
import os
import json
from tqdm import tqdm
from openai import OpenAI


def match_answer_with_gpt(decoded_output: str, options: dict, openai_api_key: str) -> str:
    """
    Use OpenAI API to determine which multiple choice option the decoded output matches.

    Args:
        decoded_output: The model's generated response
        options: Dictionary of options (e.g., {"a": "Option A text", "b": "Option B text", ...})
        openai_api_key: OpenAI API key

    Returns:
        str: The letter of the matched option (a, b, c, d) or "?" if no match
    """
    client = OpenAI(api_key=openai_api_key)

    # Build the prompt for GPT
    options_text = "\n".join([f"{label}) {text}" for label, text in options.items()])

    prompt = f"""You are evaluating a model's answer to a multiple-choice question.
                The model provided the following response:
                "{decoded_output}"

                The available options are:
                {options_text}

                Your task is to determine which option (a, b, c, or d) the model's response best matches or refers to.
                - If the response clearly indicates or matches one of the options, return ONLY the letter (a, b, c, or d).
                - If the response doesn't match any option or is ambiguous, return ONLY "?".

                Return ONLY a single character: a, b, c, d, or ?
                Do not provide any explanation."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
            messages=[
                {"role": "system", "content": "You are a precise answer matching assistant. Return only a single character."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1
        )

        predicted = response.choices[0].message.content.strip().lower()

        # Validate the response
        if predicted in ["a", "b", "c", "d", "?"]:
            return predicted
        else:
            return "?"

    except Exception as e:
        print(f"⚠️  OpenAI API error: {e}")
        return "?"


def benchmark_exam(model, tokenizer, data_path, preprompt, max_new_tokens=20, temperature=0.0, openai_api_key=None):
    with open(data_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    if temperature>0:
        sample=True
    else:
        sample=False

    print(f"✅ Loaded {len(questions)} questions from {data_path}")
    
    correct = 0
    results = []

    for q in tqdm(questions):
        q_text = f"\nQuestion: {q['question']}\n"
        for label, option in q['options'].items():
            q_text += f"{label}) {option}\n"
        q_text += "Answer:"

        messages = [
            {"role": "system", "content": preprompt},
            {"role": "user", "content": q_text}
        ]

        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        output_tokens = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        input_len = tokenized_chat.shape[-1]
        new_tokens = output_tokens[0, input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Use GPT-based matching if API key is provided, otherwise use simple method
        if openai_api_key:
            predicted = match_answer_with_gpt(decoded, q['options'], openai_api_key)
        else:
            predicted = decoded[0] if decoded and decoded[0] in ["a", "b", "c", "d"] else "?"

        results.append({
            "id": q["id"],
            "question": q["question"],
            "predicted_raw": decoded,
            "predicted_answer": predicted,
            "correct": q["correct_answer"],
            "is_correct": predicted == q["correct_answer"]
        })

        if predicted == q["correct_answer"]:
            correct += 1
        print(f"Question {q['id']}: Raw='{decoded}' → Predicted={predicted} (Correct={q['correct_answer']})")

    accuracy = correct / len(questions)
    print(f"\n📊 Accuracy: {accuracy:.2%} ({correct}/{len(questions)})")

    exam_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = f"../results/benchmark_results_{exam_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📝 Results saved to {output_file}")
    return accuracy, results


 
# # example usage 
# preprompt_file = "/users/klleshi/LLMProject/Latin-chatbot/data/prepromt_French.txt"
# exam_path = "/users/klleshi/LLMProject/Latin-chatbot/data/Olivier/2025_06_Benchmark_Olivier.json"
# with open(preprompt_file, 'r', encoding='utf-8') as f:
#         preprompt = f.read()
# benchmark_exam(
#         model=model,
#         tokenizer=tokenizer,
#         data_path=exam_path,
#         preprompt=preprompt,
#     )
