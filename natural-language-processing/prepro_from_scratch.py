def remove_symbols(file_path):
    symbols = "!\"£$%^&*()\',./#;:[]"
    print(file_path)
    print(file_path[:-3])

    with open(file_path, "r") as base_file:
        lines = base_file.readlines()
        for index, _ in enumerate(lines):
            for symbol in symbols:
                lines[index] = lines[index].replace(symbol,"")
    
    with open(file_path[:-4] + "-nosym.txt", "w") as processed_file:
        processed_file.writelines(lines)


if __name__ == "__main__":
    remove_symbols("natural-language-processing/data/poems.txt")