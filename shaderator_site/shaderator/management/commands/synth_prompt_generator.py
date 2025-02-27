#synth_prompt_generator.py
import openai
import os
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from shaderator.models import SandboxGLSLCorpus

class Command(BaseCommand):
    help = 'Generate GPT prompts for each loaded GLSL code'

    def handle(self, *args, **kwargs):
        
        # shaders = SandboxGLSLCorpus.objects.all()
        shaders = SandboxGLSLCorpus.objects.filter(id__gt=37743)
        
        properties_path = os.path.join(settings.BASE_DIR, 'properties.json')

        # Define a test variable to break the loop
        test = 0
        with open(properties_path, encoding='utf-8') as p:
            props = json.load(p)

        # Set your OpenAI API key here, it is obtained from the properties.json file
        openai.api_key = props['open_ai_key']

        # Get default model from properties
        default_model = props['default_model']

        # Prepare the prompt for the model
        prompt = [
            {"role": "system", "content": ' You are a fragment shader expert\
                                            and understand what a fragment shader is rendering \
                                            by just looking at the code.\
                                            You can describe what the code is rendering in simple terms, emphazising the \
                                            visual aspects of it.\
                                            You respond briefly and with no mention of technical details.\
                                            Do not start the response with "the shader" or "this shader" or "the code",\
                                            just with the visual description without referencing the person who asked.'},
            {"role": "user", "content": ''}]
        for shader in shaders:
            prompt[1]['content'] = 'what does the following GLSL shader code renders: \n' + shader.code
            # print(prompt[1]['content'])
            try:
                completion = openai.ChatCompletion.create(
                    model=default_model,
                    messages=prompt,
                    # messages=f"Describe a prompt that could generate the following GLSL shader code:\n\n{shader.code}",
                    max_tokens=150
                )
                
                shader.synth_prompt = completion.choices[0].message['content']
                shader.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully generated prompt for {shader.id}'))
                # test += 1
                # if test > 20:
                #     break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error generating prompt for {shader.id}: {e}'))
                break