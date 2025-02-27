import json
import time
import requests
from django.core.management.base import BaseCommand
from shaderator.models import Shader, Renderpass, Fetch_Errors

class Command(BaseCommand):
    help = 'Fetch and store shader code from Shadertoy'

    def handle(self, *args, **kwargs):
        shaders = Shader.objects.all()
        api_key = 'BdrtMR'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # Open a log file in append mode
        with open('fetch_shaders.log', 'a') as log_file:

            for shader in shaders:
                error = False
                renderpass_url = f"https://www.shadertoy.com/api/v1/shaders/{shader.shader_id}?key={api_key}"
                
                try:
                    response = requests.get(renderpass_url, headers=headers)
                    response.raise_for_status()
                except requests.exceptions.HTTPError as http_err:
                    error = True
                    if response.status_code == 403:
                        log_file.write(f'403 Forbidden: Invalid API key or insufficient permissions for shader ID {shader.shader_id}\n')
                    else:
                        log_file.write(f'HTTP error occurred for shader ID {shader.shader_id}: {http_err}\n')
                    continue
                except requests.exceptions.RequestException as e:
                    error = True
                    log_file.write(f'Request error occurred for shader ID {shader.shader_id}: {e}\n')
                    continue

                try:
                    data_shader = response.json()
                    renderpasses = data_shader.get('Shader', {}).get('renderpass', [])
                    
                    for renderpass_data in renderpasses:
                        r = Renderpass()
                        r.code = renderpass_data.get('code', '')
                        r.shader_id = shader  # 'shader' is a ForeignKey in Renderpass model
                        
                        try:
                            r.save()
                            log_file.write(f'Successfully saved renderpass for shader ID {shader.shader_id}\n')
                        except Exception as e:
                            error = True
                            log_file.write(f'Error saving renderpass for shader ID {shader.shader_id}: {e}\n')
                except (json.JSONDecodeError, KeyError) as e:
                    error = True
                    log_file.write(f'Error parsing JSON response for shader ID {shader.shader_id}: {e}\n')
                if error:
                    fe = Fetch_Errors()
                    fe.shader_id = shader
                    try:
                        fe.save()
                        log_file.write(f'Successfully saved fetch error for shader ID {shader.shader_id}\n')
                    except Exception as e:
                        log_file.write(f'Error saving fetch error for shader ID {shader.shader_id}: {e}\n')
                # Rate limiting: sleep for 1 second between requests
                time.sleep(1)
