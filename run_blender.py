import argparse
import subprocess
import os
import re
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

INFERENCE = os.path.join(PROJECT_DIR, "inference.py")

BLENDER_SCRIPT = os.path.join(
    PROJECT_DIR,
    "blender",
    "generated_script.py"
)

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def generate_code(prompt):
    cmd = [
        sys.executable,
        INFERENCE,
        "--prompt",
        prompt
    ]

    print("Using Python:", sys.executable)
    print("Running:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        return None

    output = result.stdout

    start = output.find("GENERATED CODE:")
    if start == -1:
        return None

    code = output[start + len("GENERATED CODE:"):]
    code = code.replace("=" * 60, "")
    code = re.sub(r"<\|.*?\|>", "", code)

    return code.strip()

def save_script(code):
    os.makedirs(os.path.dirname(BLENDER_SCRIPT), exist_ok=True)

    with open(BLENDER_SCRIPT, "w", encoding="utf-8") as f:
        f.write(code)


def launch_blender():
    subprocess.run(
        [
            BLENDER_EXE,
            "--python",
            BLENDER_SCRIPT
        ],
        cwd=PROJECT_DIR
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    print("Generating Blender code...\n")

    code = generate_code(args.prompt)

    if code is None:
        print("Generation failed.")
        return

    print(code)

    save_script(code)

    print("\nLaunching Blender...\n")

    launch_blender()


if __name__ == "__main__":
    main()