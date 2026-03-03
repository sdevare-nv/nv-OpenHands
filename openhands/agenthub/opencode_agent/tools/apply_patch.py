from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

APPLY_PATCH_TOOL_NAME = 'apply_patch'

_APPLY_PATCH_DESCRIPTION = """Applies a unified diff patch to modify multiple files in a single operation.

Usage:
- Apply changes to multiple files atomically
- Supports add, update, delete, and move operations
- Uses standard unified diff format
- All changes are validated before being applied
- LSP diagnostics are run after applying changes

Parameters:
- patchText: The full patch text in unified diff format that describes all changes

Patch Format:
The patch must be enclosed in *** Begin Patch and *** End Patch markers.

For adding a new file:
*** Begin Patch
--- /dev/null
+++ path/to/new/file.py
@@ -0,0 +1,3 @@
+def hello():
+    print("Hello")
+
*** End Patch

For updating an existing file:
*** Begin Patch
--- path/to/file.py
+++ path/to/file.py
@@ -10,7 +10,7 @@
 unchanged line
-old line
+new line
 unchanged line
*** End Patch

For deleting a file:
*** Begin Patch
--- path/to/file.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line 1
-line 2
-line 3
*** End Patch

For moving/renaming a file with changes:
*** Begin Patch
--- old/path/file.py
+++ new/path/file.py
@@ -1,3 +1,3 @@
 unchanged line
-old line
+new line
*** End Patch

Examples:
- Add new file: patchText with +++ for new file path
- Update file: patchText with changes to existing file
- Delete file: patchText with +++ /dev/null
- Move file: patchText with different --- and +++ paths

Note: This tool is useful for making multiple related changes across files in a single atomic operation."""

ApplyPatchTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=APPLY_PATCH_TOOL_NAME,
        description=_APPLY_PATCH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['patchText'],
            'properties': {
                'patchText': {
                    'type': 'string',
                    'description': 'The full patch text that describes all changes to be made',
                },
            },
        },
    ),
)
