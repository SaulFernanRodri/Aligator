def read(path):
    try:
        with open(path, 'r', encoding='utf-8') as txt_file:
            content = txt_file.read()
            print(content)

    except FileNotFoundError:
        print(f"El archivo en la ruta '{path}' no fue encontrado.")

    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return None
