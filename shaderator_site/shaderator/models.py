from django.db import models
from random import randint
import numpy as np

# Create your models here.

# Shader table with each shader
class Shader(models.Model):
    id = models.AutoField(primary_key=True)
    shader_id = models.CharField(max_length=10,default='shader_id',unique=True)
    # shader_id = models.CharField(max_length=10,unique=True)
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    username = models.CharField(max_length=200)
    
    def get_description(self):
        return self.description
   
    def get_name(self):
        return self.name
    
    def get_username(self):
        return self.username
    
    def get_id(self):
        return self.id
    
    def get_shader_id(self):
        return self.shader_id

    def get_random():
        max_id = Shader.objects.count()
        print(max_id)
        while True:
            num = randint(1, max_id)
            print(num)
            shader = Shader.objects.filter(pk=num).first()
            print(shader.shader_id)
            if shader:
                return shader

class Prompt(models.Model):
    HUMAN = 'human'
    SYNTHETIC = 'synthetic'
    USER_GENERATED = 'user'

    SOURCE_CHOICES = [
        (HUMAN, 'Human'), # The shadercorpus prompts
        (SYNTHETIC, 'Synthetic'), # The glsl sandbox prompts
        (USER_GENERATED, 'User Generated'), # The shadergen prompts
    ]

    text = models.TextField(unique=True)
    vector = models.BinaryField(null=True, blank=True)  # Store the vector as a binary field
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=USER_GENERATED, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def get_prompt(self):
        return self.text

    def save_vector(self, vector):
        self.vector = vector.tobytes()

    def load_vector(self):
        return np.frombuffer(self.vector, dtype=np.float32) if self.vector else None

    def __str__(self):
        return f"{self.get_source_display()}: {self.text[:50]}"

class GeneratedShaderCode(models.Model):
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE, null=True, blank=True)
    # user_prompt = models.TextField()
    code = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    # vector = models.BinaryField(null=True, blank=True)  # Store the vector as a binary field
    rating = models.IntegerField(null=True, blank=True)

    # def get_user_prompt(self):
    #     return self.user_prompt
    
    def get_prompt(self):
        return self.prompt

    def get_code(self):
        return self.code

    def get_rating(self):
        return self.rating

    # def save_vector(self, vector):
    #     self.vector = vector.tobytes()

    # def load_vector(self):
    #     return np.frombuffer(self.vector, dtype=np.float32) if self.vector else None

# The renderpass table, it will contain each renderpass of the shaders
class Renderpass(models.Model):
    shader_id = models.ForeignKey(Shader, on_delete=models.CASCADE,default='shader_id')
    renderpass_id = models.IntegerField(primary_key=True)
    code = models.TextField()

# The Fetch Errors table will contain each shader id with errors at fetch
class Fetch_Errors(models.Model):
    shader_id = models.ForeignKey(Shader, on_delete=models.CASCADE,default='shader_id')


# Integration with shadescriptor

# The user's session info
class User_Session(models.Model):
    user_id = models.AutoField(primary_key=True)
    session_key = models.CharField(max_length=40)
    shader_familiarity = models.BooleanField(null=True)
    cscience_familiarity = models.BooleanField(null=True)
    programming_familiarity = models.BooleanField(null=True)
    new_media_familiarity = models.BooleanField(null=True)

    def get_user_id(self):
        return self.user_id
    
    def get_session_key(self):
        return self.session_key
 
class ShadescriptorCorpus(models.Model):
    id = models.AutoField(primary_key=True)
    shader_id = models.CharField(max_length=10)
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE, null=True, blank=True)
    # human_prompt = models.TextField()
    code = models.TextField()
    # vector = models.BinaryField(null=True, blank=True)  # Store the vector as a binary field
    
    # shadescriptor integration
    # input_date = models.DateTimeField('date of input',null=True)
    user_id = models.ForeignKey(User_Session,to_field='user_id', on_delete=models.CASCADE, blank=True, null=True)

    # def get_human_prompt(self):
    #     return self.human_prompt
    
    def get_prompt(self):
        return self.prompt

    def get_code(self):
        return self.code

    # def save_vector(self, vector):
    #     self.vector = vector.tobytes()

    # def load_vector(self):
    #     return np.frombuffer(self.vector, dtype=np.float32) if self.vector else None
    
class SandboxGLSLCorpus(models.Model):
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE, null=True, blank=True)
    # synth_prompt = models.TextField()
    code = models.TextField()
    # vector = models.BinaryField(null=True, blank=True)  # Store the vector as a binary field

    # def get_synth_prompt(self):
    #     return self.synth_prompt
    
    def get_prompt(self):
        return self.prompt
    
    def get_code(self):
        return self.code

    # def save_vector(self, vector):
    #     self.vector = vector.tobytes()

    # def load_vector(self):
    #     return np.frombuffer(self.vector, dtype=np.float32) if self.vector else None

class ShaderGenerationRecord(models.Model):
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE)
    code = models.TextField()  # The generated shader code
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp of when the generation was created
    compiled_successfully = models.BooleanField(default=False)  # Whether the shader compiled successfully
    error_message = models.TextField(null=True, blank=True)  # Any error messages if compilation failed
    rating = models.IntegerField(null=True, blank=True)  # User rating, if applicable

    def get_prompt(self):
        return self.prompt

    def get_code(self):
        return self.code

    def get_rating(self):
        return self.rating

    def __str__(self):
        return f"ShaderGenerationRecord {self.id} - Compiled: {self.compiled_successfully}"

class ShaderModelGenerationRecord(models.Model):
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=50)  # The name of the GPT model used
    code = models.TextField()  # The generated shader code
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp of when the generation was created
    compiled_successfully = models.BooleanField(default=False)  # Whether the shader compiled successfully
    error_message = models.TextField(null=True, blank=True)  # Error message if compilation failed
    attempt_number = models.IntegerField()  # The attempt number for this record
    rating = models.IntegerField(null=True, blank=True)  # User rating, if applicable

    def get_prompt(self):
        return self.prompt

    def get_code(self):
        return self.code

    def get_model_name(self):
        return self.model_name

    def get_rating(self):
        return self.rating

    def __str__(self):
        return f"ShaderModelGenerationRecord {self.id} - Model: {self.model_name} - Attempt: {self.attempt_number} - Compiled: {self.compiled_successfully}"
