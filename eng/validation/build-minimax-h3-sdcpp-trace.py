#!/usr/bin/env python3
"""Link an owned eval-callback wrapper using an already built, unchanged SD CLI.

Only output files are written. Existing upstream sources, objects and libraries
are consumed unchanged, including the original CLI source with renamed main.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build=args.build.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    commands=json.loads((build/'compile_commands.json').read_text())
    original=next(item for item in commands if item['file'].endswith('/examples/cli/main.cpp'))
    source=Path(original['file'])
    wrapper=Path(__file__).with_name('minimax-h3-sdcpp-trace.cpp').resolve()
    command=shlex.split(original['command'])
    obj=output/'trace.o'
    command[command.index('-o')+1]=str(obj)
    command=[str(wrapper) if value==str(source) else value for value in command]
    command.insert(1,'-DSD_CLI_SOURCE="'+str(source)+'"')
    # Include the original CLI's adjacent image_metadata.h without copying it.
    command.insert(1,'-I'+str(source.parent))
    subprocess.run(command,cwd=original['directory'],check=True)
    directory=build/'examples/cli'
    link=shlex.split((directory/'CMakeFiles/sd-cli.dir/link.txt').read_text())
    link=[str(obj) if value.endswith('CMakeFiles/sd-cli.dir/main.cpp.o') else value for value in link]
    binary=output/'sd-cli-trace'
    link[link.index('-o')+1]=str(binary)
    subprocess.run(link,cwd=directory,check=True)
    record={'original_cli_source':str(source),'original_cli_source_sha256':sha(source),
            'wrapper_sha256':sha(wrapper),'binary':str(binary),'binary_sha256':sha(binary),
            'compile_argv':command,'link_argv':link,
            'timing_limitation':'Evaluation callback synchronizes diagnostic boundaries; never use this binary for timing'}
    (output/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'binary':str(binary),'sha256':record['binary_sha256']}))


if __name__=='__main__':
    main()
