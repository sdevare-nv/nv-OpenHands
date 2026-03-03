"""
Exact Python implementations of OpenCode tools.
These implementations match the TypeScript OpenCode implementations.
"""

import os
import re
from pathlib import Path
from typing import Generator

# ============================================================================
# Constants
# ============================================================================

DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024  # 50KB

BINARY_EXTENSIONS = {
    '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.class', '.jar', '.war',
    '.7z', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods',
    '.odp', '.bin', '.dat', '.obj', '.o', '.a', '.lib', '.wasm', '.pyc', '.pyo'
}

IGNORE_PATTERNS = [
    'node_modules/', '__pycache__/', '.git/', 'dist/', 'build/', 'target/',
    'vendor/', 'bin/', 'obj/', '.idea/', '.vscode/', '.zig-cache/', 'zig-out',
    '.coverage', 'coverage/', 'tmp/', 'temp/', '.cache/', 'cache/', 'logs/',
    '.venv/', 'venv/', 'env/'
]

# Similarity thresholds for block anchor fallback matching
SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.0
MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.3

# ============================================================================
# Read Tool Implementation
# ============================================================================


def is_binary_file(filepath: str) -> bool:
    """Check if a file is binary."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return True

    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(4096)
            if not chunk:
                return False

            # Check for null bytes
            if b'\x00' in chunk:
                return True

            # Count non-printable characters
            non_printable = sum(
                1 for byte in chunk
                if byte < 9 or (byte > 13 and byte < 32)
            )

            # If >30% non-printable characters, consider it binary
            return non_printable / len(chunk) > 0.3
    except Exception:
        return False


def read_file_opencode(
    filepath: str,
    offset: int = 0,
    limit: int = DEFAULT_READ_LIMIT
) -> str:
    """
    Read a file with OpenCode-style formatting.
    Returns formatted output with line numbers (5-digit padded).
    """
    # Make path absolute
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    # Check if file exists
    if not os.path.exists(filepath):
        # Try to find suggestions
        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        if os.path.isdir(directory):
            entries = os.listdir(directory)
            suggestions = [
                os.path.join(directory, entry)
                for entry in entries
                if basename.lower() in entry.lower() or entry.lower() in basename.lower()
            ][:3]

            if suggestions:
                return f"File not found: {filepath}\n\nDid you mean one of these?\n" + "\n".join(suggestions)

        return f"File not found: {filepath}"

    # Check if binary
    if is_binary_file(filepath):
        return f"Cannot read binary file: {filepath}"

    # Read file
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().split('\n')

    # Process lines with offset and limit
    raw = []
    total_bytes = 0
    truncated_by_bytes = False

    for i in range(offset, min(len(lines), offset + limit)):
        line = lines[i]
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "..."

        line_bytes = len(line.encode('utf-8')) + (1 if raw else 0)
        if total_bytes + line_bytes > MAX_BYTES:
            truncated_by_bytes = True
            break

        raw.append(line)
        total_bytes += line_bytes

    # Format with line numbers (5-digit padded with |)
    content = [
        f"{str(i + offset + 1).zfill(5)}| {line}"
        for i, line in enumerate(raw)
    ]

    total_lines = len(lines)
    last_read_line = offset + len(raw)
    has_more_lines = total_lines > last_read_line
    truncated = has_more_lines or truncated_by_bytes

    output = "<file>\n"
    output += "\n".join(content)

    if truncated_by_bytes:
        output += f"\n\n(Output truncated at {MAX_BYTES} bytes. Use 'offset' parameter to read beyond line {last_read_line})"
    elif has_more_lines:
        output += f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {last_read_line})"
    else:
        output += f"\n\n(End of file - total {total_lines} lines)"

    output += "\n</file>"
    return output


# ============================================================================
# Edit Tool Implementation - Fuzzy Matching Replacers
# ============================================================================


def levenshtein(a: str, b: str) -> int:
    """Levenshtein distance algorithm implementation."""
    if a == "" or b == "":
        return max(len(a), len(b))

    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )

    return matrix[len(a)][len(b)]


def simple_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Simple exact match replacer."""
    yield find


def line_trimmed_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match by trimmed lines."""
    original_lines = content.split('\n')
    search_lines = find.split('\n')

    if search_lines and search_lines[-1] == '':
        search_lines.pop()

    for i in range(len(original_lines) - len(search_lines) + 1):
        matches = True

        for j in range(len(search_lines)):
            original_trimmed = original_lines[i + j].strip()
            search_trimmed = search_lines[j].strip()

            if original_trimmed != search_trimmed:
                matches = False
                break

        if matches:
            match_start_index = sum(len(original_lines[k]) + 1 for k in range(i))
            match_end_index = match_start_index
            for k in range(len(search_lines)):
                match_end_index += len(original_lines[i + k])
                if k < len(search_lines) - 1:
                    match_end_index += 1

            yield content[match_start_index:match_end_index]


def block_anchor_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match by first and last line anchors with similarity check."""
    original_lines = content.split('\n')
    search_lines = find.split('\n')

    if len(search_lines) < 3:
        return

    if search_lines and search_lines[-1] == '':
        search_lines.pop()

    first_line_search = search_lines[0].strip()
    last_line_search = search_lines[-1].strip()
    search_block_size = len(search_lines)

    # Collect all candidate positions
    candidates = []
    for i in range(len(original_lines)):
        if original_lines[i].strip() != first_line_search:
            continue

        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_line_search:
                candidates.append({'start_line': i, 'end_line': j})
                break

    if not candidates:
        return

    # Handle single candidate
    if len(candidates) == 1:
        start_line = candidates[0]['start_line']
        end_line = candidates[0]['end_line']
        actual_block_size = end_line - start_line + 1

        similarity = 0
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check > 0:
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                original_line = original_lines[start_line + j].strip()
                search_line = search_lines[j].strip()
                max_len = max(len(original_line), len(search_line))
                if max_len == 0:
                    continue
                distance = levenshtein(original_line, search_line)
                similarity += (1 - distance / max_len) / lines_to_check

                if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
                    break
        else:
            similarity = 1.0

        if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            match_start_index = sum(len(original_lines[k]) + 1 for k in range(start_line))
            match_end_index = match_start_index
            for k in range(start_line, end_line + 1):
                match_end_index += len(original_lines[k])
                if k < end_line:
                    match_end_index += 1
            yield content[match_start_index:match_end_index]
        return

    # Multiple candidates - find best match
    best_match = None
    max_similarity = -1

    for candidate in candidates:
        start_line = candidate['start_line']
        end_line = candidate['end_line']
        actual_block_size = end_line - start_line + 1

        similarity = 0
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check > 0:
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                original_line = original_lines[start_line + j].strip()
                search_line = search_lines[j].strip()
                max_len = max(len(original_line), len(search_line))
                if max_len == 0:
                    continue
                distance = levenshtein(original_line, search_line)
                similarity += 1 - distance / max_len
            similarity /= lines_to_check
        else:
            similarity = 1.0

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = candidate

    if max_similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD and best_match:
        start_line = best_match['start_line']
        end_line = best_match['end_line']
        match_start_index = sum(len(original_lines[k]) + 1 for k in range(start_line))
        match_end_index = match_start_index
        for k in range(start_line, end_line + 1):
            match_end_index += len(original_lines[k])
            if k < end_line:
                match_end_index += 1
        yield content[match_start_index:match_end_index]


def whitespace_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with normalized whitespace."""
    def normalize_whitespace(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    normalized_find = normalize_whitespace(find)
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if normalize_whitespace(line) == normalized_find:
            yield line
        else:
            normalized_line = normalize_whitespace(line)
            if normalized_find in normalized_line:
                words = find.strip().split()
                if words:
                    pattern = r'\s+'.join(re.escape(word) for word in words)
                    try:
                        match = re.search(pattern, line)
                        if match:
                            yield match.group(0)
                    except re.error:
                        pass

    # Handle multi-line matches
    find_lines = find.split('\n')
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = lines[i:i + len(find_lines)]
            if normalize_whitespace('\n'.join(block)) == normalized_find:
                yield '\n'.join(block)


def indentation_flexible_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with flexible indentation."""
    def remove_indentation(text: str) -> str:
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return text

        min_indent = min(
            len(l) - len(l.lstrip())
            for l in non_empty_lines
        )

        return '\n'.join(
            line if not line.strip() else line[min_indent:]
            for line in lines
        )

    normalized_find = remove_indentation(find)
    content_lines = content.split('\n')
    find_lines = find.split('\n')

    for i in range(len(content_lines) - len(find_lines) + 1):
        block = '\n'.join(content_lines[i:i + len(find_lines)])
        if remove_indentation(block) == normalized_find:
            yield block


def escape_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with normalized escape sequences."""
    def unescape_string(s: str) -> str:
        replacements = {
            '\\n': '\n', '\\t': '\t', '\\r': '\r',
            "\\'": "'", '\\"': '"', '\\`': '`',
            '\\\\': '\\', '\\$': '$'
        }
        for escaped, unescaped in replacements.items():
            s = s.replace(escaped, unescaped)
        return s

    unescaped_find = unescape_string(find)

    if unescaped_find in content:
        yield unescaped_find

    lines = content.split('\n')
    find_lines = unescaped_find.split('\n')

    for i in range(len(lines) - len(find_lines) + 1):
        block = '\n'.join(lines[i:i + len(find_lines)])
        unescaped_block = unescape_string(block)

        if unescaped_block == unescaped_find:
            yield block


def multi_occurrence_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Yield all exact matches."""
    start_index = 0
    while True:
        index = content.find(find, start_index)
        if index == -1:
            break
        yield find
        start_index = index + len(find)


def trimmed_boundary_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with trimmed boundaries."""
    trimmed_find = find.strip()

    if trimmed_find == find:
        return

    if trimmed_find in content:
        yield trimmed_find

    lines = content.split('\n')
    find_lines = find.split('\n')

    for i in range(len(lines) - len(find_lines) + 1):
        block = '\n'.join(lines[i:i + len(find_lines)])
        if block.strip() == trimmed_find:
            yield block


def context_aware_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match using context anchors."""
    find_lines = find.split('\n')
    if len(find_lines) < 3:
        return

    if find_lines and find_lines[-1] == '':
        find_lines.pop()

    content_lines = content.split('\n')
    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()

    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue

        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                block_lines = content_lines[i:j + 1]
                block = '\n'.join(block_lines)

                if len(block_lines) == len(find_lines):
                    matching_lines = 0
                    total_non_empty = 0

                    for k in range(1, len(block_lines) - 1):
                        block_line = block_lines[k].strip()
                        find_line = find_lines[k].strip()

                        if block_line or find_line:
                            total_non_empty += 1
                            if block_line == find_line:
                                matching_lines += 1

                    if total_non_empty == 0 or matching_lines / total_non_empty >= 0.5:
                        yield block
                        break
                break


def replace_with_fuzzy_matching(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> str:
    """
    Replace old_string with new_string using fuzzy matching.
    Tries multiple replacers in order until a match is found.
    """
    if old_string == new_string:
        raise ValueError("oldString and newString must be different")

    not_found = True

    replacers = [
        simple_replacer,
        line_trimmed_replacer,
        block_anchor_replacer,
        whitespace_normalized_replacer,
        indentation_flexible_replacer,
        escape_normalized_replacer,
        trimmed_boundary_replacer,
        context_aware_replacer,
        multi_occurrence_replacer,
    ]

    for replacer in replacers:
        for search in replacer(content, old_string):
            index = content.find(search)
            if index == -1:
                continue

            not_found = False

            if replace_all:
                return content.replace(search, new_string)

            last_index = content.rfind(search)
            if index != last_index:
                continue

            return content[:index] + new_string + content[index + len(search):]

    if not_found:
        raise ValueError("oldString not found in content")

    raise ValueError(
        "Found multiple matches for oldString. Provide more surrounding lines "
        "in oldString to identify the correct match."
    )


def edit_file_opencode(
    filepath: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> str:
    """
    Edit a file with OpenCode-style fuzzy matching.
    Returns status message.
    """
    # Make path absolute
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    # Handle empty old_string (create/overwrite file)
    if old_string == "":
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_string)
        return "Edit applied successfully (file created/overwritten)."

    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    if os.path.isdir(filepath):
        raise IsADirectoryError(f"Path is a directory, not a file: {filepath}")

    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply replacement with fuzzy matching
    new_content = replace_with_fuzzy_matching(content, old_string, new_string, replace_all)

    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return "Edit applied successfully."


# ============================================================================
# Glob Tool Implementation
# ============================================================================


def glob_files_opencode(pattern: str, search_path: str = '.') -> str:
    """
    Find files matching a glob pattern, sorted by modification time.
    Returns formatted output.
    """
    import fnmatch

    # Make path absolute
    if not os.path.isabs(search_path):
        search_path = os.path.join(os.getcwd(), search_path)

    if not os.path.isdir(search_path):
        return f"Directory not found: {search_path}"

    limit = 100
    files = []
    truncated = False

    # Walk directory and match files
    for root, _, filenames in os.walk(search_path):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                full_path = os.path.join(root, filename)
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    mtime = 0
                files.append({'path': full_path, 'mtime': mtime})

                if len(files) >= limit:
                    truncated = True
                    break

        if truncated:
            break

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['mtime'], reverse=True)

    # Build output
    if not files:
        return "No files found"

    output = [f['path'] for f in files]

    if truncated:
        output.append("")
        output.append("(Results are truncated. Consider using a more specific path or pattern.)")

    return '\n'.join(output)


# ============================================================================
# List Tool Implementation
# ============================================================================


def list_dir_opencode(search_path: str = '.', ignore_patterns: list = None) -> str:
    """
    List directory contents in tree structure.
    Returns formatted output.
    """
    if ignore_patterns is None:
        ignore_patterns = IGNORE_PATTERNS

    # Make path absolute
    if not os.path.isabs(search_path):
        search_path = os.path.join(os.getcwd(), search_path)

    if not os.path.isdir(search_path):
        return f"Directory not found: {search_path}"

    limit = 100
    files = []

    # Collect files
    for root, dirs, filenames in os.walk(search_path):
        # Filter out ignored directories
        rel_root = os.path.relpath(root, search_path)
        skip = False
        for pattern in ignore_patterns:
            pattern_clean = pattern.rstrip('/')
            if pattern_clean in rel_root or rel_root.startswith(pattern_clean):
                skip = True
                break

        if skip:
            dirs[:] = []  # Don't descend into ignored directories
            continue

        # Filter directories in-place
        dirs[:] = [
            d for d in dirs
            if not any(
                d == p.rstrip('/') or d.startswith(p.rstrip('/'))
                for p in ignore_patterns
            )
        ]

        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(root, filename), search_path)
            files.append(rel_path)
            if len(files) >= limit:
                break

        if len(files) >= limit:
            break

    # Build directory structure
    dirs_set = set()
    files_by_dir = {}

    for file in files:
        dir_path = os.path.dirname(file)
        parts = dir_path.split(os.sep) if dir_path != '.' and dir_path else []

        # Add all parent directories
        for i in range(len(parts) + 1):
            dir_p = os.sep.join(parts[:i]) if i > 0 else '.'
            dirs_set.add(dir_p)

        # Add file to its directory
        dir_key = dir_path if dir_path else '.'
        if dir_key not in files_by_dir:
            files_by_dir[dir_key] = []
        files_by_dir[dir_key].append(os.path.basename(file))

    def render_dir(dir_path: str, depth: int) -> str:
        indent = '  ' * depth
        output = ''

        if depth > 0:
            output += f"{indent}{os.path.basename(dir_path)}/\n"

        child_indent = '  ' * (depth + 1)

        # Get child directories
        children = sorted([
            d for d in dirs_set
            if os.path.dirname(d) == dir_path and d != dir_path
        ])

        # Render subdirectories first
        for child in children:
            output += render_dir(child, depth + 1)

        # Render files
        dir_files = sorted(files_by_dir.get(dir_path, []))
        for f in dir_files:
            output += f"{child_indent}{f}\n"

        return output

    output = f"{search_path}/\n" + render_dir('.', 0)
    return output


# ============================================================================
# Write Tool Implementation
# ============================================================================


def write_file_opencode(filepath: str, content: str) -> str:
    """
    Write content to a file.
    Returns status message.
    """
    # Make path absolute
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return "Wrote file successfully."

