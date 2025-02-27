from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.management import call_command
from django.utils import timezone
from django.shortcuts import redirect

# import urllib library
from urllib.request import urlopen 
# import json
import json
# import os for os level settings
import os
# openai API
from openai import OpenAI
# import numpy
import numpy as np
# import faiss
import faiss
# import sencence transformer
from sentence_transformers import SentenceTransformer
# My imports.
from .models import Shader, Renderpass, GeneratedShaderCode, ShadescriptorCorpus, SandboxGLSLCorpus, User_Session
from .models import Prompt, ShaderGenerationRecord

import re

# for corpus download
import csv

# Create your views here.

def index(request):
    # return HttpResponse("Hello, world. You're at the shaderator index.")
    return render(request, 'shaderator_site/intro.html')

def preprocess_shader_code(generated_code):
    print('Preprocessing shader code...', flush=True)
    
    # Remove any existing precision declarations
    generated_code = re.sub(r'precision\s+(lowp|mediump|highp)\s+float;\n?', '', generated_code)
    
    # Start with the precision qualifier at the very beginning
    final_code = "precision highp float;\n"
    
    # Add placeholders for missing utility functions if needed
    # if 'rand' in generated_code:
    #     rand_function = """
    #     float rand(vec2 co) {
    #         return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
    #     }
    #     """
    #     final_code += rand_function

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

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
def initialize_faiss_index(corpus_model):
    # Retrieve all related prompts with non-null vectors
    corpus = corpus_model.objects.exclude(prompt__vector__isnull=True).select_related('prompt')

    # Extract vectors from the related Prompt model
    vectors = [item.prompt.load_vector() for item in corpus if item.prompt and item.prompt.load_vector() is not None]
    
    if vectors:
        dimension = len(vectors[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(vectors, dtype=np.float32))
        return faiss_index, corpus
    
    return None, None

# Initialize FAISS indices for the three different corpora
human_faiss_index, human_corpus = initialize_faiss_index(ShadescriptorCorpus)
synthetic_faiss_index, synthetic_corpus = initialize_faiss_index(SandboxGLSLCorpus)
generated_faiss_index, generated_corpus = initialize_faiss_index(GeneratedShaderCode)


def retrieve_examples(query, faiss_index, corpus):
    # Encode the query to get the vector
    query_vector = model.encode([query])[0]
    
    # Perform the FAISS search to find the closest matches
    indices = faiss_index.search(np.array([query_vector], dtype=np.float32), k=3)[1][0]
    indices = [int(idx) for idx in indices]
    
    # Retrieve the prompts and codes from the corpus using the indices
    prompts = [corpus[idx].prompt.text for idx in indices]
    codes = [corpus[idx].get_code() for idx in indices]
    
    return zip(prompts, codes)

def retrieve_generated_examples(query, faiss_index, corpus):
    query_vector = model.encode([query])[0]
    indices = faiss_index.search(np.array([query_vector], dtype=np.float32), k=3)[1][0]
    indices = [int(idx) for idx in indices]

    # Retrieve prompts, codes, and ratings
    # examples = [(corpus[idx].prompt.text, corpus[idx].get_code(), corpus[idx].get_rating()) for idx in indices]
    # Retrieve prompts, codes, and ratings and print out the ratings
    examples = []
    for idx in indices:
        rating = corpus[idx].get_rating()
        # print(f"Rating for index {idx}: {rating}")  # Print the rating
        examples.append((corpus[idx].prompt.text, corpus[idx].get_code(), rating))

    # Sort by rating in descending order and take the top 3
    sorted_examples = sorted(examples, key=lambda x: x[2], reverse=True)[:1]
    
    # Unpack the sorted examples into prompts and codes
    prompts, codes, _ = zip(*sorted_examples)
    
    return zip(prompts, codes)

def search_shader_descriptions(request):
    results = []
    if request.method == 'POST':
        query_description = request.POST.get('prompt', '')
        query_vector = model.encode([query_description])[0]
        # indices = faiss_index.search(np.array([query_vector], dtype=np.float32), k=3)[1][0]

        indices = human_faiss_index.search(np.array([query_vector], dtype=np.float32), k=3)[1][0]

        # Convert FAISS indices to regular Python integers
        indices = [int(idx) for idx in indices]

        for index in indices:
            shader = human_corpus[index]
            results.append((shader.id, shader.get_prompt().text))

        indices = synthetic_faiss_index.search(np.array([query_vector], dtype=np.float32), k=3)[1][0]

        # Convert FAISS indices to regular Python integers
        indices = [int(idx) for idx in indices]

        for index in indices:
            shader = synthetic_corpus[index]
            results.append((shader.id, shader.get_prompt().text))

        indices = generated_faiss_index.search(np.array([query_vector], dtype=np.float32), k=3)[1][0]

        # Convert FAISS indices to regular Python integers
        indices = [int(idx) for idx in indices]

        for index in indices:
            shader = generated_corpus[index]
            results.append((shader.id, shader.get_prompt().text))

    return render(request, 'shaderator_site/search.html', {'results': results})

def download_corpus(request):
    """Generate and return a CSV file with the full shader corpus."""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="shader_corpus.csv"'

    writer = csv.writer(response)
    writer.writerow(['ID', 'Source', 'Description', 'Code'])

    # Combine all corpora into one list
    # full_corpus = human_corpus + synthetic_corpus + generated_corpus
    full_corpus = list(human_corpus) + list(synthetic_corpus) + list(generated_corpus)

    for shader in full_corpus:
        # Use shader.prompt.source instead of shader.source
        writer.writerow([shader.id, shader.prompt.source, shader.get_prompt().text, shader.get_code()])

    # for shader in full_corpus:
    #     writer.writerow([shader.id, shader.source, shader.get_prompt().text])

    return response

def generate_shader_code(prompt, human_examples, synthetic_examples, generated_examples):
    print("generate_shader_code", flush=True)

    # Load properties from the JSON file
    properties_path = os.path.join(settings.BASE_DIR, 'properties.json')
    with open(properties_path, encoding='utf-8') as p:
        props = json.load(p)

    print("Setting OpenAI key...", flush=True)
    
    # Set your OpenAI API key here, it is obtained from the properties.json file
    client = OpenAI(api_key=props['open_ai_key'])
    
    print("Getting default properties...", flush=True)
    # Get default model from properties
    default_model = props['default_model']
    # default_model = props['fine_tuned_model_en']

    # Construct the augmented prompt with examples
    augmented_prompt = (
        "Generate shader code for the following User Prompt, using the examples below as guidance:\n\n"
        f"### User Prompt:\n{prompt}\n\n"
    )

    # Convert zips to lists before slicing
    human_examples = list(human_examples)
    synthetic_examples = list(synthetic_examples)
    generated_examples = list(generated_examples)

    # Add Human Example
    if human_examples:
        for example, code in human_examples[:1]:  # Taking the first example
            augmented_prompt += (
                f"### Example 1 (Human):\n"
                f"Prompt: {example}\n"
                f"Code:\n{code}\n\n"
            )

    # Add Synthetic Example
    if synthetic_examples:
        for example, code in synthetic_examples[:1]:  # Taking the first example
            augmented_prompt += (
                f"### Example 2 (Synthetic):\n"
                f"Prompt: {example}\n"
                f"Code:\n{code}\n\n"
            )

    # Add Generated Example
    if generated_examples:
        for example, code in generated_examples[:1]:  # Taking the first example
            augmented_prompt += (
                f"### Example 3 (Generated):\n"
                f"Prompt: {example}\n"
                f"Code:\n{code}\n\n"
            )

    # Prepare the prompt for the model
    prompt = [
        {
            "role": "system",
            "content": (
                "You respond to requests with plain Shadertoy or GLSL shader code, without any words that aren't part of the code. "
                "Do not use ```glsl at the start and end of the response. "
                "Do not include uniform definitions nor the main function. "
                "Use the provided examples only as guidelines for style and structure. Do not base your response directly on these examples. "
                "The generated code should be independently tailored to the new user prompt. "
                "Ensure all variable types are consistent and correctly defined. For example, `vec3` should only be assigned to `vec3` values, and `vec2` should only be assigned to `vec2` values. "
                "Avoid type mismatches such as assigning a `vec3` to a `vec2` variable or vice versa. "
                "Double-check all assignments and operations for type compatibility."
            )
        },
        {
            "role": "user",
            "content": augmented_prompt
        }
    ]




    print('Passed Prompt: ', flush=True)
    print(prompt, flush=True)
    
    # Call the OpenAI API to generate shader code
    try:
        print('Calling OpenAI API...', flush=True)
        completion = client.chat.completions.create(
            model=default_model,
            messages=prompt,
            temperature=1.0
        )
        # Correct way to access the content in the response
        response_message = completion.choices[0].message.content

        shader_code = preprocess_shader_code(response_message)
        print('Shader code received: ', flush=True)
        print(shader_code, flush=True)
        return shader_code

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}", flush=True)
        return None




def shader_view(request):
    print("shader_view", flush=True)
    if request.method == 'POST':
        data = json.loads(request.body)
        prompt_text = data.get('prompt', '')
        print("prompt received: " + prompt_text, flush=True)

        # Get or create prompt in the database
        prompt_vector = model.encode(prompt_text)
        new_prompt, created = Prompt.objects.get_or_create(
            text=prompt_text,
            defaults={'vector': prompt_vector, 'source': 'user'}
        )

        # Retrieve relevant examples from both corpora
        human_examples = retrieve_examples(prompt_text, human_faiss_index, human_corpus)
        synthetic_examples = retrieve_examples(prompt_text, synthetic_faiss_index, synthetic_corpus)
        generated_examples = retrieve_generated_examples(prompt_text, generated_faiss_index, generated_corpus)

        # Generate shader code using the augmented prompt
        shader_code = generate_shader_code(prompt_text, human_examples, synthetic_examples, generated_examples)
        if shader_code:
            print('retorna codigo de shader', flush=True)
            return JsonResponse({'shader_code': shader_code, 'prompt_id': new_prompt.id})
        else:
            return JsonResponse({'error': 'Failed to generate shader code'}, status=500)
    else:
        print("else shader_view", flush=True)
        return render(request, 'shaderator_site/shader.html')
    
@csrf_exempt
def store_shader_view(request):
    print('saving generated code in database...', flush=True)
    if request.method == 'POST':
        data = json.loads(request.body)
        shader_code = data.get('shader_code', '')        
        shader_rating = data.get('rating','')
        prompt_data = data.get('prompt')

        # Get the prompt from database
        # new_prompt = Prompt.objects.get(id = prompt_id)
        prompt = Prompt.objects.get(text=prompt_data)
        print(prompt)
        # Store the shader code in the database
        new_shader = GeneratedShaderCode(prompt=prompt,code=shader_code, rating=shader_rating)
        new_shader.save()

        return JsonResponse({'status': 'success', 'message': 'Shader code stored successfully'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

@csrf_exempt  # Ensure to use CSRF exemption only if necessary and secure the view accordingly
def store_shader_record(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt_id = data.get('prompt_id')
            code = data.get('code')
            compiled_successfully = data.get('compiled_successfully', False)
            error_message = data.get('error_message', '')

            prompt = Prompt.objects.get(id=prompt_id)

            # Create and save the ShaderGenerationRecord
            shader_record = ShaderGenerationRecord.objects.create(
                prompt=prompt,
                code=code,
                compiled_successfully=compiled_successfully,
                error_message=error_message
            )

            return JsonResponse({'status': 'success', 'message': 'Shader compilation record stored successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def load_shader_code(request):

    if request.method=="POST":
        
        for shader in Shader:
            
            renderpass_url = "https://www.shadertoy.com/api/v1/shaders/{}?key=BdrtMR".format(shader.shader_id)

            try:
                shader_response = urlopen(renderpass_url)
            except:
                print('there was an error trying to open the shadertoy url', flush=True)
                print('shader id: ' + str(shader.shader_id),flush=True)
                print(str(IOError),flush=True)
                return render(request, 'shaderator_site/index.html')
                            
            data_shader = json.loads(shader_response.read())
            render_obj = data_shader['Shader']['renderpass']
            for i in render_obj:
                r = Renderpass()
                r.code = i['code']
                r.shader_id = shader
                try:
                    r.save()
                except:
                    print('error intentando grabar renderpass', flush=True)
                
            return render(request, 'shaderator_site/thanks.html')
    else:
        return render(request, 'shaderator_site/index.html')
    
def run_management_command(request):
    try:
        # Call your management command here
        call_command('generate_few_shot', 'trained')
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
# Shadescriptor Integration:
def home(request):
    print('estoy en casa', flush=True)
    print('intento recuperar usuario de la base de datos', flush=True)
    if not request.session.session_key:
        request.session.save()
    print('session ' , request.session.session_key, flush=True)
    try:
        user = User_Session.objects.get(session_key = request.session.session_key)
        print('id del usuario en la base', user.session_key, flush=True)
    except:
        print('el usuario no existe', flush=True)
        user = User_Session()

    print(user.user_id, flush=True)
    if user.user_id is None:
        print('intento grabar sesion de usuario', flush=True)
        # request.session.save()
        user.session_key = request.session.session_key
        print('grabo sesion de usuario:', flush=True)
        print(user.session_key, flush=True)
        try:
            user.save()
        except:
            print('error intentando grabar el usuario en la base de datos', flush=True)
    context = {
                'user' : user
    }
    return render(request, 'shaderator_site/home.html', context)

def mobile_modal(request):
    print('abro html para mobile en lugar del modal', flush=True)
    return render(request, 'shaderator_site/mobileModal.html')

def shaders(request):
    if not request.session or not request.session.session_key:
        return home(request)
    shader = Shader.get_random()
    context = {
                'id' : shader.id,
                'random_shader': shader.shader_id,
                'shader_name' : shader.name,
                'shader_desc' : shader.description,
                'shader_username' : shader.username
                }
    return render(request, 'shaderator_site/index.html',context)

def store_input(request):

    if request.method=="POST":

        #get shader info from db via post request
        shader_id = request.POST['shader_id']

        # get renderpass of shader from renderpass model
        shader = Shader.objects.get(id = shader_id)
        renderpass_url = "https://www.shadertoy.com/api/v1/shaders/{}?key=BdrtMR".format(shader.shader_id)

        try:
            shader_response = urlopen(renderpass_url)
        except:
            print('hubo un error al intentar abrir la url de shadertoy', flush=True)
            print('shader id: ' + str(shader_id),flush=True)
            print(str(IOError),flush=True)
            return shaders(request)
                
        data_shader = json.loads(shader_response.read())
        render_obj = data_shader['Shader']['renderpass']

        # add to corpus for each code obtained from shadertoy
        for i in render_obj:
            #create prompt object
            prompt = Prompt()
            shadescriptor = ShadescriptorCorpus()
            shadescriptor.shader_id = Shader.objects.get(pk = shader_id)
            #get input text from html object
            prompt.text=request.POST['paragraph_text']
            #get date of input
            prompt.created_at = timezone.now()
            # shadescriptor.input_date = timezone.now()
            shadescriptor.code = i['code']
            prompt.shader_id = shader
            prompt.source = Prompt.HUMAN
            #get user id from the session
            print("busco el usuario de clave:", flush=True)
            print(request.session.session_key, flush=True)
            user = User_Session.objects.get(session_key = request.session.session_key)
            shadescriptor.user_id = user
            # vectorize prompt and store it
            prompt.vector = model.encode(prompt.text)
            shadescriptor.prompt = prompt
            try:
                prompt.save()
                shadescriptor.save()
            except:
                print('Failed to store input in database', flush=True)

        return render(request, 'shaderator_site/thanks.html')
    else:
        return shaders(request)
    

def store_user_info(request):
    if request.method=="POST":
        user = User_Session.objects.get(session_key = request.session.session_key)
        user.shader_familiarity = request.POST['familiarity']
        user.cscience_familiarity = request.POST['csience']
        user.programming_familiarity = request.POST['programming']
        user.new_media_familiarity = request.POST['nma']
        try:
            user.save()
        except:
            print('error intentando guardar usuario en la base de datos', flush=True)
        # Redirect to 'shaders' after successful form submission
        return redirect('shaders')
    else:
        return redirect('home')

    #     return shaders(request)
    # else:
    #     return shaders(request)