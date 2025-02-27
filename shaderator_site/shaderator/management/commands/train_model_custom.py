import json
import re
from collections import defaultdict

from django.core.management.base import BaseCommand
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from datasets import Dataset
from shaderator.tokenizer import ShaderTokenizer

class Command(BaseCommand):
    help = 'Tokenize shader code and fine-tune GPT-2 model'

    def handle(self, *args, **options):

        # Load the tokenized dataset
        def load_dataset(file_path, tokenizer):
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=128
            )
        
        def preprocess_dataset(dataset):
            tokenizer = ShaderTokenizer()
            tokenized_dataset = []
            tokens = tokenizer.tokenize(dataset)
            tokenized_dataset.extend(tokens)
            print(tokenized_dataset)
            return tokenized_dataset
        
        # Example shader code for testing
   
        # shader_code = """
        # void main() {
        #     vec3 color = vec3(1.0, 0.0, 0.0); // Red color
        #     gl_FragColor = vec4(color, 1.0);
        # }
        # """

        shader_code = """
        void mainImage( out vec4 fragColor, in vec2 fragCoord )
        {
            vec2 uv = fragCoord/iResolution.xy;
            vec3 col = vec3(0.0);
            
            // moving fire
            float t = iTime * 2.0;
            for(float i = 0.0; i < 8.0; i += 1.0) {
                float fi = fract(i/8.0 + sin(t)/3.0);
                float v = 1.0 - smoothstep(0.0, 0.5, length(uv - vec2(0.5 + sin(i + t) * 0.1, fi * 0.7)));
                col += vec3(v, v * (0.2 + sin(t)*0.3), v * 0.1);
            }

            // smoke effect
            col = mix(col, vec3(0.5), smoothstep(0.0, 1.0, uv.y));

            fragColor = vec4(col, 1.0);
        }
        """
        # Verify GPU availability
        print("GPU Available: ", torch.cuda.is_available())
        print("GPU Name: ", torch.cuda.get_device_name(0))

        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        tokenized_dataset = preprocess_dataset(shader_code)
        if not tokenized_dataset:
            print("Error Tokenizing...")
            quit()
        else:
            print('Tokenization Successfull.') 

        # Remove the comments:
        for token in tokenized_dataset:
            if token[0] == '\n':
                tokenized_dataset.remove(token)

        # print(tokenized_dataset)

        # Save the tokenized dataset to a file
        with open("tokenized_dataset.txt", "w") as f:
            for token in tokenized_dataset:
                f.write(f"{token}\n")

        tokenized_file_path = "tokenized_dataset.txt"
        train_dataset = load_dataset(tokenized_file_path, tokenizer)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Initialize GPT-2 model
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        training_args = TrainingArguments(
            output_dir="./results/gpt2-shader-trained",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # Train the model
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained("./results/gpt2-shader-trained")
        tokenizer.save_pretrained("./results/gpt2-shader-trained")