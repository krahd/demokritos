from django.core.management.base import BaseCommand
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import keyboard

def wait_for_enter_or_exit():
    print("Press Enter to continue or any other key to exit...")
    
    while True:
        event = keyboard.read_event()
        
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'enter':
                print("Continuing the program...")
                break
            else:
                print("Exiting the program...")
                exit()

class Command(BaseCommand):
    help = 'Train the GPT-2 model on shader code'

    def handle(self, *args, **kwargs):
        # Verify GPU availability
        print("GPU Available: ", torch.cuda.is_available())
        print("GPU Name: ", torch.cuda.get_device_name(0))

        wait_for_enter_or_exit()
        
        # Load pre-trained model and tokenizer
        model_name = "gpt2"  # Use a smaller model to speed up training such as gpt2-small or gpt2-medium
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare dataset
        def load_dataset(file_path, tokenizer, block_size=128):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().split('---')  # Use '---' as delimiter if needed
            texts = [line.strip() for line in lines if line.strip()]

            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=block_size,
                overwrite_cache=True
            )
            return dataset

        dataset = load_dataset("shader_code.txt", tokenizer)

        # Create a data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=1,  # Reduce number of epochs
            per_device_train_batch_size=4,  # Increase batch size if possible
            save_steps=10_000,
            save_total_limit=2,
            fp16=True,  # Enable mixed precision training
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Train the model
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained("./results")
        tokenizer.save_pretrained("./results")

        self.stdout.write(self.style.SUCCESS("Model and tokenizer saved to ./results"))

        # Generate Shader Code
        # from transformers import pipeline

        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

        # prompt = "void mainImage( out vec4 fragColor, in vec2 fragCoord ) {"
        # generated_code = generator(prompt, max_length=200, num_return_sequences=1)

        # print(generated_code[0]['generated_text'])