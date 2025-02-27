import numpy as np
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from shaderator.models import ShadescriptorCorpus, SandboxGLSLCorpus, GeneratedShaderCode, Prompt
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Load shader descriptions from the database and generate vectors'

    def handle(self, *args, **kwargs):
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # # Query the ShadescriptorCorpus table for descriptions
        # shader_corpus = ShadescriptorCorpus.objects.all()
        # shader_descriptions = [shader.get_prompt() for shader in shader_corpus]

        # # Check if descriptions are correctly fetched
        # if not all(isinstance(desc, str) for desc in shader_descriptions):
        #     self.stdout.write(self.style.ERROR('Some descriptions are not strings.'))
        #     return

        # # Convert descriptions to embeddings using the pre-trained model
        # vectors = model.encode(shader_descriptions)

        # # Save the vectors back to the database
        # for shader, vector in tqdm(zip(shader_corpus, vectors), total=len(shader_corpus), desc="Processing"):
        #     shader.save_vector(np.array(vector, dtype=np.float32))
        #     shader.save()

        # # Query the ShaderDescriptorCorpus table for descriptions
        # shader_corpus = SandboxGLSLCorpus.objects.all()
        # shader_descriptions = [shader.get_prompt() for shader in shader_corpus]

        # # Check if descriptions are correctly fetched
        # if not all(isinstance(desc, str) for desc in shader_descriptions):
        #     self.stdout.write(self.style.ERROR('Some descriptions are not strings.'))
        #     return

        # # Convert descriptions to embeddings using the pre-trained model
        # vectors = model.encode(shader_descriptions)

        # # Save the vectors back to the database
        # for shader, vector in tqdm(zip(shader_corpus, vectors), total=len(shader_corpus), desc="Processing"):
        #     shader.save_vector(np.array(vector, dtype=np.float32))
        #     shader.save()

        # self.stdout.write(self.style.SUCCESS('Successfully loaded shader descriptions and generated vectors'))

        # Fetch all GeneratedShaderCode entries
        shader_corpus = GeneratedShaderCode.objects.all()

        # Query the ShaderDescriptorCorpus table for descriptions
        shader_descriptions = [shader.get_prompt().text for shader in shader_corpus]

        # Check if descriptions are correctly fetched
        if not all(isinstance(desc, str) for desc in shader_descriptions):
            self.stdout.write(self.style.ERROR('Some descriptions are not strings.'))
            return

        # Convert descriptions to embeddings using the pre-trained model
        vectors = model.encode(shader_descriptions)

        # Save the vectors back to the Prompt model
        for shader, vector in tqdm(zip(shader_corpus, vectors), total=len(shader_corpus), desc="Processing"):
            prompt = shader.get_prompt()  # Get the related prompt object
            if prompt and prompt.source == Prompt.USER_GENERATED:
                prompt.vector = np.array(vector, dtype=np.float32).tobytes()
                prompt.save()
            else:
                self.stdout.write(self.style.WARNING(f"Prompt not found or source is not user-generated for shader ID {shader.id}"))

        self.stdout.write(self.style.SUCCESS('Successfully loaded shader descriptions and generated vectors'))