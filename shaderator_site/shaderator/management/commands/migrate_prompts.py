from django.core.management.base import BaseCommand
from shaderator.models import ShadescriptorCorpus, SandboxGLSLCorpus, GeneratedShaderCode, Prompt

class Command(BaseCommand):
    help = 'Migrate existing prompts to the new Prompt model with correct source attribution'

    def handle(self, *args, **kwargs):
        # Migrate Human Corpus Prompts
        for shader in ShadescriptorCorpus.objects.all():
            prompt_text = shader.get_human_prompt()
            if not prompt_text:  # Skip if the prompt is empty or None
                self.stdout.write(f"Skipping empty or null Human prompt for shader ID: {shader.id}")
                continue
            
            prompt_vector = shader.load_vector()
            prompt, created = Prompt.objects.get_or_create(
                text=prompt_text,
                defaults={'vector': prompt_vector, 'source': Prompt.HUMAN}
            )
            shader.prompt = prompt
            shader.save()
            if created:
                self.stdout.write(f"Created Human prompt: {prompt_text[:50]}")

        # Migrate Synthetic Corpus Prompts
        for shader in SandboxGLSLCorpus.objects.all():
            prompt_text = shader.get_synth_prompt()
            if not prompt_text:  # Skip if the prompt is empty or None
                self.stdout.write(f"Skipping empty or null Synthetic prompt for shader ID: {shader.id}")
                continue
            
            prompt_vector = shader.load_vector()
            prompt, created = Prompt.objects.get_or_create(
                text=prompt_text,
                defaults={'vector': prompt_vector, 'source': Prompt.SYNTHETIC}
            )
            shader.prompt = prompt
            shader.save()
            if created:
                self.stdout.write(f"Created Synthetic prompt: {prompt_text[:50]}")

        # Migrate User Generated Prompts
        for shader in GeneratedShaderCode.objects.all():
            prompt_text = shader.get_user_prompt()
            if not prompt_text:  # Skip if the prompt is empty or None
                self.stdout.write(f"Skipping empty or null User Generated prompt for shader ID: {shader.id}")
                continue
            
            prompt_vector = shader.load_vector()
            prompt, created = Prompt.objects.get_or_create(
                text=prompt_text,
                defaults={'vector': prompt_vector, 'source': Prompt.USER_GENERATED}
            )
            shader.prompt = prompt
            shader.save()
            if created:
                self.stdout.write(f"Created User Generated prompt: {prompt_text[:50]}")

        self.stdout.write(self.style.SUCCESS('Successfully migrated all prompts to the Prompt model.'))
