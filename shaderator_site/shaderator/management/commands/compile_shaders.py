import subprocess
import re
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db.models import Subquery
# My imports.
from shaderator.models import Prompt, ShaderModelGenerationRecord
# openai API
from openai import OpenAI
import anthropic
import json, os

class Command(BaseCommand):
    help = 'Generate and compile shader code for user prompts with multiple models'

    def handle(self, *args, **kwargs):
        # Load properties from the JSON file
        properties_path = os.path.join(settings.BASE_DIR, 'properties.json')
        with open(properties_path, encoding='utf-8') as p:
            props = json.load(p)

        print("Setting API keys...", flush=True)
        
        # Load API keys and model information
        open_ai_key = props['open_ai_key']
        default_model = props['default_model']
        fine_tuned_model = props['fine_tuned_model_en']
        default_model_3_5 = props['default_model_3_5']
        claude_ai_key = props['claude_ai_key']
        claude_model = props['claude_model']

        # Initialize OpenAI and Claude clients
        openai_client = OpenAI(api_key=open_ai_key)
        claude_client = anthropic.Anthropic(api_key=claude_ai_key)

        # Get the IDs of prompts that are already in the ShaderModelGenerationRecord model
        existing_prompt_ids = ShaderModelGenerationRecord.objects.values_list('prompt_id', flat=True)

        # Fetch prompts where the source is 'user' and the prompt ID is not in existing_prompt_ids
        user_prompts = Prompt.objects.filter(source='user').exclude(id__in=Subquery(existing_prompt_ids))

        # Dictionary of models and their respective names
        models = {
            'gpt-4': default_model,
            'gpt-3.5': default_model_3_5,
            'gpt-3.5-fine-tuned': fine_tuned_model,
            'claude.ai': claude_model
        }

        # Iterate over each prompt and process them
        for prompt in user_prompts:
            prompt_text = prompt.text
            self.stdout.write(f"Processing prompt: {prompt_text}")

            # For each model, try to generate and compile shader code with retries
            for model_name, model in models.items():

                self.compile_shader_code_with_retries(prompt, openai_client, claude_client, model_name, model)

            # just one execution for debugging
            # break

    def generate_shader_code_for_model(self, prompt_text, openai_client, claude_client, model_name, model):
        """
        This function generates shader code for a specific model (GPT or Claude).
        """
        # Create the appropriate prompt
        augmented_prompt, system_prompt = self.create_prompt(prompt_text)
        
        if model_name == 'claude.ai':
            claude_prompt = [{"role": "user", "content": prompt_text}]
            try:
                response = claude_client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=claude_prompt
                )
                return response.content[0].text
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to generate shader code: {str(e)}"))
        else:
            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=augmented_prompt,
                    temperature=1.0  # Adjust as needed
                )
                return response.choices[0].message.content
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to generate shader code: {str(e)}"))
        
    def create_prompt(self, prompt_text):
        """
        Create the augmented prompt for each GPT model
        """
        system_prompt = "You respond to requests with plain Shadertoy or GLSL shader code, without any words that aren't part of the code. "\
                    "Do not use ```glsl at the start and end of the response. "\
                    "Do not include uniform definitions nor the main function. "\
                    "Use the provided examples only as guidelines for style and structure. Do not base your response directly on these examples. "\
                    "Ensure all variable types are consistent and correctly defined. For example, `vec3` should only be assigned to `vec3` values, and `vec2` should only be assigned to `vec2` values. "\
                    "Avoid type mismatches such as assigning a `vec3` to a `vec2` variable or vice versa. "\
                    "Double-check all assignments and operations for type compatibility."
        
        augmented_prompt = f"Generate shader code for: {prompt_text}"
        prompt = [
            {
                "role": "system",
                "content": ( system_prompt
                    # "You respond to requests with plain Shadertoy or GLSL shader code, without any words that aren't part of the code. "
                    # "Do not use ```glsl at the start and end of the response. "
                    # "Do not include uniform definitions nor the main function. "
                    # "Use the provided examples only as guidelines for style and structure. Do not base your response directly on these examples. "
                    # "Ensure all variable types are consistent and correctly defined. For example, `vec3` should only be assigned to `vec3` values, and `vec2` should only be assigned to `vec2` values. "
                    # "Avoid type mismatches such as assigning a `vec3` to a `vec2` variable or vice versa. "
                    # "Double-check all assignments and operations for type compatibility."
                )
            },
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
        return prompt, system_prompt

    def preprocess_shader_code(self, generated_code):
        print('Preprocessing shader code...', flush=True)
        
        # Remove any existing precision declarations
        generated_code = re.sub(r'precision\s+(lowp|mediump|highp)\s+float;\n?', '', generated_code)
        
        # Start with the precision qualifier at the very beginning
        final_code = "precision highp float;\n"

        # Check for Shadertoy-specific uniforms or main function    
        contains_resolution = re.search(r'uniform\s+vec2\s+iResolution', generated_code) is not None
        contains_time = re.search(r'uniform\s+float\s+iTime', generated_code) is not None
        contains_mouse = re.search(r'uniform\s+vec2\s+iMouse', generated_code) is not None
        contains_main = re.search(r'void\s+main\s*\(\s*\)', generated_code) is not None
        
        # Add Shadertoy uniforms only if they are not already present
        if not contains_resolution:
            final_code += "uniform vec2 iResolution;\n"
        if not contains_time:
            final_code += "uniform float iTime;\n"
        if not contains_mouse:
            final_code += "uniform vec2 iMouse;\n"
        
        # Add the generated code
        final_code += generated_code
        
        # If the generated code doesn't have a main function, wrap it in one
        if not contains_main:
            final_code += 'void main() {\n    mainImage(gl_FragColor, gl_FragCoord.xy);\n}\n'
        
        # Adjust loop indices to use constants if needed and avoid type mismatches
        final_code = re.sub(
            r'for\s*\(\s*int\s+(\w+)\s*=\s*(\d+);\s*\1\s*<\s*(\d+\.?\d*);\s*\1\s*\+\+\s*\)', 
            r'for (int \1 = \2; \1 < int(\3); \1++)', 
            final_code
        )
        
        # Ensure vector assignments are correct (e.g., avoid assigning vec3 to vec2)
        def fix_vector_assignments(code):
            assignment_pattern = re.compile(r'(vec[2|3]\s+\w+\s*=\s*)vec(\d)\s*\(([^)]+)\);')
            
            def replace_match(match):
                var_declaration = match.group(1)  # e.g., 'vec2 color = '
                source_vec_type = match.group(2)  # e.g., '3' if vec3
                components = match.group(3)       # e.g., '1.0, 2.0, 3.0'
                
                target_vec_size = int(var_declaration.split()[0][-1])  # vec2 or vec3
                source_vec_size = int(source_vec_type)  # 2 or 3
                
                if source_vec_size > target_vec_size:
                    # Drop extra components
                    new_components = ', '.join(components.split(',')[:target_vec_size])
                elif source_vec_size < target_vec_size:
                    # Add default components (e.g., 0.0)
                    new_components = ', '.join(components.split(',')) + ', ' + ', '.join(['0.0'] * (target_vec_size - source_vec_size))
                else:
                    new_components = components
                
                return f"{var_declaration}vec{target_vec_size}({new_components});"
            
            return assignment_pattern.sub(replace_match, code)

        final_code = fix_vector_assignments(final_code)

        return final_code

    def compile_shader_code_with_retries(self, prompt, openai_client, claude_client, model_name, model):
        """
        Function to generate and compile GLSL code up to 3 times, generating new code on each attempt.
        """
        compiled_successfully = False
        
        for attempt in range(1, 4):  # 3 attempts
            # Generate new shader code for each attempt
            shader_code = self.generate_shader_code_for_model(prompt.text, openai_client, claude_client, model_name, model)

            # Preprocess the shader code
            preprocessed_code = self.preprocess_shader_code(shader_code)

            # Compile the shader code
            compiled_successfully, error_message = self.compile_shader_code(preprocessed_code)

            # Save the result of this attempt in the database
            ShaderModelGenerationRecord.objects.create(
                prompt=prompt,
                model_name=model_name,
                code=preprocessed_code,
                compiled_successfully=compiled_successfully,
                error_message=error_message if not compiled_successfully else None,
                attempt_number=attempt
            )

            if compiled_successfully:
                break  # Exit if compilation is successful

    def compile_shader_code(self, shader_code, shader_stage='frag'):
        """
        Compile the shader code using glslangValidator and log the results.
        Args:
            shader_code: The shader code to compile.
            shader_stage: The stage of the shader ('frag' for fragment, 'vert' for vertex, etc.).
        """
        try:
            # Write shader code to a temporary file with the appropriate extension based on shader stage
            temp_shader_filename = f'temp/temp_shader.{shader_stage}.glsl'
            with open(temp_shader_filename, 'w') as f:
                f.write(shader_code)

            # Run glslangValidator to compile the shader, specifying the shader stage if needed
            result = subprocess.run(['glslangValidator', temp_shader_filename], capture_output=True, text=True)
            
            # Append the result to the compilation log
            with open('temp/compilation_log.txt', 'a') as log_file:
                log_file.write("Shader Compilation Attempt:\n")
                log_file.write(f"Shader Code:\n{shader_code}\n")
                log_file.write(f"Compilation Output:\n{result.stdout}\n")
                log_file.write(f"Compilation Errors:\n{result.stderr}\n")
                log_file.write("========================================\n")

            # Check if the compilation was successful
            if result.returncode == 0:
                return True, None  # Successfully compiled
            else:
                return False, result.stdout  # Compilation failed, return error message
        except Exception as e:
            # Log the exception as well
            with open('temp/compilation_log.txt', 'a') as log_file:
                log_file.write(f"Exception during compilation: {str(e)}\n")
                log_file.write("========================================\n")
            return False, str(e)

