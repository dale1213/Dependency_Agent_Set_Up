import json

def convert_to_kumu_format(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize elements and connections lists
    elements = []
    connections = []
    
    # Add all models as elements
    for model_id in data.keys():
        elements.append({
            "id": f"model_{model_id.replace('/', '_')}",
            "label": model_id,
            "type": "model"
        })
    
    # Add all packages as elements and create connections
    for model_id, model_data in data.items():
        if model_data:  # If model has packages
            for combo in model_data:
                for package, version in combo.items():
                    # Add package as element if not already added
                    package_id = f"package_{package}=={version}"
                    if not any(e["id"] == package_id for e in elements):
                        elements.append({
                            "id": package_id,
                            "label": f"{package}=={version}",
                            "type": "package"
                        })
                    
                    # Create connection between model and package
                    connections.append({
                        "from": f"model_{model_id.replace('/', '_')}",
                        "to": package_id
                    })
    
    # Create the final Kumu format
    kumu_data = {
        "elements": elements,
        "connections": connections
    }
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(kumu_data, f, indent=2)

def main():
    # Convert all iteration files
    for i in range(1, 7):
        input_file = f'successful_combinations {i}.json'
        output_file = f'plot/kumu_format_{i}.json'
        convert_to_kumu_format(input_file, output_file)
        print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    main() 