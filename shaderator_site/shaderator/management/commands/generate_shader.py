import json
from django.core.management.base import BaseCommand
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

class Command(BaseCommand):
    help = 'Generate shader code using the fine-tuned GPT-2 model'

    def add_arguments(self, parser):
        parser.add_argument('prompt', type=str, help='The input prompt for shader generation')

    def handle(self, *args, **kwargs):
        prompt = kwargs['prompt']

        # Correct path to the fine-tuned model and tokenizer
        # model_path = os.path.join(os.path.dirname(__file__), '../../../results/gpt2-shader-trained')
        model_path = os.path.join(os.path.dirname(__file__), '../../../results')
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            self.stdout.write(self.style.ERROR(f"Model path does not exist: {model_path}"))
            return

        try:
            # Load the fine-tuned model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to load model or tokenizer: {str(e)}"))
            return

        # Encode the input prompt
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to tokenize input prompt: {str(e)}"))
            return

        # Generate shader code
        try:
            outputs = model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to generate shader code: {str(e)}"))
            return

        # Decode the generated code
        try:
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to decode generated code: {str(e)}"))
            return

        self.stdout.write(self.style.SUCCESS(f'Generated Shader Code:\n{generated_code}'))
