from pathlib import Path


def build_tree(path: Path):
    """
    Recursively build a directory tree using pathlib.
    - Directories have: name, type='directory', children=[...], __size__, __count__
    - Files have: name, type='file', __size__, __count__ (no separate 'size')
    """
    if path.is_dir():
        tree = {
            "name": path.name or str(path),
            "type": "directory",
            "children": [],
            "__size__": 0,
            "__count__": 0
        }

        for child in sorted(path.iterdir(), key=lambda p: p.name):
            child_tree = build_tree(child)
            tree["children"].append(child_tree)

            # accumulate totals from children
            tree["__size__"] += child_tree["__size__"]
            tree["__count__"] += child_tree["__count__"]

        return tree

    else:
        try:
            size = path.stat().st_size
        except (OSError, FileNotFoundError, PermissionError):
            # Handle broken symlinks, missing files, or permission issues
            size = 0
        return {
            "name": path.name,
            "type": "file",
            "__size__": size,
            "__count__": 1
        }
    

def build_tree_from_paths(paths: list[Path], root: Path) -> dict:
    """
    Build a nested dict tree from a list of file paths, relative to a given root folder.
    Each node has:
      - __size__ : size in bytes
      - __count__: total number of files in that node
    Directories are dicts with 'name', 'type', 'children', __size__, __count__.
    Files are leaf dicts with 'name', 'type', __size__, __count__ = 1
    """
    root = Path(root).resolve()
    tree = {
        "name": root.name or str(root),
        "type": "directory",
        "children": [],
        "__size__": 0,
        "__count__": 0
    }

    # Helper: recursively insert a file path into the tree
    def insert_path(node, parts, full_path):
        if len(parts) == 1:
            # Leaf node = file
            size = full_path.stat().st_size if full_path.exists() else 0
            file_node = {
                "name": parts[0],
                "type": "file",
                "__size__": size,
                "__count__": 1
            }
            node["children"].append(file_node)
            # Update parent totals
            node["__size__"] += size
            node["__count__"] += 1
        else:
            # Directory node
            dirname = parts[0]
            # Check if child directory already exists
            child = next((c for c in node["children"] if c["type"] == "directory" and c["name"] == dirname), None)
            if child is None:
                child = {
                    "name": dirname,
                    "type": "directory",
                    "children": [],
                    "__size__": 0,
                    "__count__": 0
                }
                node["children"].append(child)
            # Recurse into remaining parts
            insert_path(child, parts[1:], full_path)
            # After recursion, update totals
            node["__size__"] += child["__size__"]
            node["__count__"] += child["__count__"]

    for p in map(Path, paths):
        p = p.resolve()
        try:
            rel_path = p.relative_to(root)
        except ValueError:
            continue  # skip files outside root
        insert_path(tree, rel_path.parts, p)

    return tree


def print_tree(tree, prefix="", max_depth=None, depth=0, is_last=True):
    """
    Recursively print a directory tree with __size__ and __count__.
    
    Args:
        tree (dict): node from build_tree() or build_tree_from_paths().
        prefix (str): string prefix for indentation.
        max_depth (int or None): maximum depth to print. None = no limit.
        depth (int): current depth (used internally).
        is_last (bool): whether this node is the last child of its parent.
    """
    connector = "└── " if is_last else "├── "
    size_human = _format_size(tree['__size__'])
    print(f"{prefix}{connector}{tree['name']} (size: {size_human}, files: {tree['__count__']})")

    if tree["type"] == "directory":
        if max_depth is not None and depth >= max_depth:
            if tree["children"]:
                print(f"{prefix}    └── ...")
            return

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(tree.get("children", [])):
            print_tree(child, prefix=child_prefix, max_depth=max_depth, depth=depth + 1, is_last=i == len(tree["children"]) - 1)

def summarize_tree(tree, max_depth=None, depth=0, parent_path=""):
    """
    Recursively summarize a tree in a table format.
    
    Args:
        tree (dict): node from build_tree() or build_tree_from_paths().
        max_depth (int or None): maximum depth to print (None = no limit).
        depth (int): current depth (used internally).
        parent_path (str): accumulated path for display
    """
    # Build the full path for display
    full_path = f"{parent_path}/{tree['name']}" if parent_path else tree['name']

    # Print the current node
    size_human = _format_size(tree['__size__'])
    print(f"{full_path:<80} | {size_human:>10} | {tree['__count__']:>8} files")

    # Stop if max_depth is reached
    if max_depth is not None and depth >= max_depth:
        return

    if tree["type"] == "directory":
        for child in sorted(tree.get("children", []), key=lambda c: (c["type"] != "directory", c["name"])):
            summarize_tree(child, max_depth=max_depth, depth=depth + 1, parent_path=full_path)

def _format_size(size_bytes: int) -> str:
    """Convert a file size in bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        size_bytes /= 1024
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
    return f"{size_bytes:.1f} PB"


if __name__ == '__main__':
    import json
    tree = build_tree(Path('/Users/moritzschmid/Code/struc2func/'))
    print_tree(tree, max_depth=3)

    summarize_tree(tree, max_depth=2)
    # print(json.dumps(tree, indent=2))
    # print('done')