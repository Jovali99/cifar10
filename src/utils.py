def print_yaml(data, indent=0):
    """Recursively print YAML."""
    spacing = "    " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}")
            print_yaml(value, indent + 1)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print(f"{spacing}- Item {index + 1}:")
            print_yaml(item, indent + 1)
    else:
        print(f"{spacing}{data}")