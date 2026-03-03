from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_APPLY_PATCH_TOOL_NAME

_APPLY_PATCH_DESCRIPTION = r"""Use the apply_patch tool to edit files.

It should be used whenever you want to create, edit, or delete a file.

IMPORTANT: NEVER, EVER use apply_patch to write tests, test files, or any code that verifies behavior. Only use it for production code, configurations, and documentation.

The patch format is a stripped-down, file-oriented diff format. The envelope is:

*** Begin Patch
[ one or more file operations ]
*** End Patch

Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

Update File may be immediately followed by *** Move to: <new path> to rename.
Then one or more "hunks", each introduced by @@ (optionally followed by a context anchor).
Within a hunk each line starts with:
  " " (space) - context line (must match the original file)
  "-" - line to remove
  "+" - line to add

Context guidelines:
- Show 3 lines of code above and below each change for context.
- If 3 lines is insufficient to uniquely identify the location, use @@ with a class or function name:
  @@ class MyClass
  @@ def my_method():

The full grammar:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ " " header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full example combining several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

Notes:
- You must include a header with your intended action (Add/Delete/Update).
- You must prefix new lines with + even when creating a new file.
- File paths must be relative, NEVER absolute.
- Multiple hunks can be included for the same file by using multiple @@ anchors.
"""

ApplyPatchTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_APPLY_PATCH_TOOL_NAME,
        description=_APPLY_PATCH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['input'],
            'properties': {
                'input': {
                    'type': 'string',
                    'description': 'The entire contents of the apply_patch command, including the *** Begin Patch and *** End Patch delimiters.',
                },
            },
            'additionalProperties': False,
        },
    ),
)
