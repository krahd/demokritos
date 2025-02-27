from django.core.management.base import BaseCommand
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

class Command(BaseCommand):
    help = 'Generate shader code using the trained GPT-2 model based on prompts'

    def add_arguments(self, parser):
        parser.add_argument('prompt', type=str, help='The prompt to generate shader code from')

    def handle(self, *args, **options):
        prompt = options['prompt']

        # Load the trained model and tokenizer
        model_name_or_path = "./results"  # Path to the directory where the model is saved
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

        # Set up the text generation pipeline
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

        # Generate shader code based on the prompt
        generated_texts = generator(prompt, max_length=1000, num_return_sequences=1)

        # Output the generated shader code
        for idx, generated_text in enumerate(generated_texts):
            self.stdout.write(self.style.SUCCESS(f"Generated Shader Code {idx+1}:"))
            self.stdout.write(generated_text['generated_text'])
