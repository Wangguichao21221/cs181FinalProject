# submit.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS181


import zipfile
from projectParams import STUDENT_CODE_DEFAULT

def zip_files(fname, files):
    with zipfile.ZipFile(fname, "w") as zipf:
        for f in files:
            zipf.write(f)

zip_files("bayesnets.zip", STUDENT_CODE_DEFAULT.split(","))
print("""Successfully generated zip file. Please submit the generated zip file to
Gradescope to receive credit.""")