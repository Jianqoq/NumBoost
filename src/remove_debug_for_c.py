import os
import re


current_path = os.getcwd()
files_and_dirs = os.listdir(current_path)
files = [f for f in files_and_dirs if os.path.isfile(
    os.path.join(current_path, f))]
c_files = [f for f in files if f.endswith('.c')]
contents = []
for f in c_files:
    with open(f, 'r') as file:
        file.seek(0)
        content = file.read()
        contents.append(content)
        new = re.sub(r'DEBUG_PRINT\(.*?\);\n', r'\n', content)
        with open(f, 'w') as file2:
            file2.write(new)

p = input(f'Press Enter after git pushed...')
for idx, f in enumerate(c_files):
    with open(f, 'w') as file:
        file.write(contents[idx])
