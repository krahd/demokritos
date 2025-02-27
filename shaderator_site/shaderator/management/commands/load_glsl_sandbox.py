# load_glsl_sandbox.py
import os
import chardet
from django.core.management.base import BaseCommand
from shaderator.models import SandboxGLSLCorpus

class Command(BaseCommand):
    help = 'Load GLSL shader files into the database'

    def add_arguments(self, parser):
        parser.add_argument('directory', type=str, help='Directory containing the GLSL shader files')

    def handle(self, *args, **kwargs):
        directory = kwargs['directory']
        loaded_files = 0
        with_errors = 0
        for filename in os.listdir(directory):
            if filename.endswith('.glsl'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'rb') as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        shader_file = SandboxGLSLCorpus(synth_prompt='', code=content, vector=None)
                        shader_file.save()
                        loaded_files += 1
                        self.stdout.write(self.style.SUCCESS(f'Successfully loaded {filename}'))
                except UnicodeDecodeError as e:
                    with_errors += 1
                    self.stdout.write(self.style.ERROR(f'Error loading {filename}: {e}'))

        self.stdout.write(self.style.SUCCESS(f'Loaded {loaded_files} files with {with_errors} errors'))