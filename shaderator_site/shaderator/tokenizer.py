import re

class ShaderTokenizer:
    def __init__(self):
        self.token_specification = [
            ('COMMENT', r'//.*?$|/\*.*?\*/'),  # Comments first
            ('KEYWORD', r'\b(?:void|float|vec2|vec3|vec4|if|else|for|while|return|out|in|iResolution|iTime)\b'),
            ('MATH_FUNCTION', r'\b(?:sin|cos|tan|mat2|mat4)\b'),
            ('QUALIFIER', r'\b(?:const|attribute|uniform|varying)\b'),
            ('OPERATOR', r'[+\-*/=<>!&|]'),
            ('PAREN', r'[()]'),  # Parentheses
            ('BRACE', r'[{}]'),  # Braces
            ('SEMICOLON', r';'),  # Semicolon
            ('COMMA', r','),  # Comma
            ('DOT', r'\.'),  # Dot
            ('IDENTIFIER', r'\b[a-zA-Z_]\w*\b'),
            ('FLOAT_LITERAL', r'\b\d+\.\d+\b'),
            ('INT_LITERAL', r'\b\d+\b'),
            ('NEWLINE', r'\n'),
            ('SKIP', r'[ \t]+'),  # Skip spaces and tabs
            ('MISMATCH', r'.'),   # Any other character
        ]
        self.token_re = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in self.token_specification)

    def tokenize(self, code):
        tokens = []
        for match in re.finditer(self.token_re, code, re.DOTALL | re.MULTILINE):
            kind = match.lastgroup
            value = match.group()
            if kind == 'NEWLINE':
                tokens.append(value)
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'{value!r} unexpected')
            else:
                tokens.append((kind, value))
        return tokens
