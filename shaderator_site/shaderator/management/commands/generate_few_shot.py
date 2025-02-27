from django.core.management.base import BaseCommand
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

from shaderator.models import GeneratedShaderCodeView

def construct_few_shot_prompt():
    # Grab examples from the generated shader code table
    examples = GeneratedShaderCodeView.objects.all()[:10]  # Fetch 5 examples
    # Generate the few-shot prompt
    prompt_template = "Example {index}:\n{prompt}\n\n{code}\n\n"
    few_shot_prompt = ""
    for index, example in enumerate(examples, start=1):
        few_shot_prompt += prompt_template.format(
            index=index,
            prompt=example.prompt,
            code=example.code
        )
    few_shot_prompt += "New Example:\n"
    return few_shot_prompt

class Command(BaseCommand):
    help = 'Generate text using the trained GPT-2 model with few-shot learning'

    def add_arguments(self, parser):
        parser.add_argument('model', type=str, help='Select the model: trained or gpt2')

    def handle(self, *args, **options):
        model_choice = options['model']
        if model_choice == 'trained':
            # Load the trained model and tokenizer
            model_name_or_path = "./results"  # Path to the directory where the model is saved
            model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Set up the text generation pipeline
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

        # Define a few-shot prompt for text generation
        few_shot_prompt = construct_few_shot_prompt()

        # Tokenize the input prompt
        input_ids = tokenizer.encode(few_shot_prompt, truncation=True, max_length=1024)
        print(f"Tokenized prompt length: {len(input_ids)}")
        print(f"Maximum token ID: {max(input_ids)}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")

        # Truncate the prompt if it's too long to ensure the total length stays within the limit
        max_prompt_length = 1024 - 500  # Assuming you want to generate up to 300 new tokens
        if len(input_ids) > max_prompt_length:
            input_ids = input_ids[:max_prompt_length]

        # Decode the truncated prompt back to text
        truncated_prompt = tokenizer.decode(input_ids)

        # Generate new text based on the truncated few-shot prompt
        generated_texts = generator(truncated_prompt, max_new_tokens=500, num_return_sequences=1)

        # Extract the generated part from the output
        print(few_shot_prompt)
        generated_text = generated_texts[0]['generated_text']
        new_generated_part = generated_text[len(truncated_prompt):].strip()

        # Print or use the generated text
        print(new_generated_part)
