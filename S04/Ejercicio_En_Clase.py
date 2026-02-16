import re  # módulo para expresiones regulares, útil para buscar patrones en texto
from collections import Counter  # estructura para contar ocurrencias de elementos

# Ejemplo sencillo de texto: cada línea es una oración
text = """<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>"""  # texto de ejemplo con marcadores de inicio/fin


# Funciones
def tokenize_sentence(sentence):  # convierte una oración en una lista de tokens
    """
    Separa una oración en "tokens" (palabras o las etiquetas <s> </s>).
    Usamos una expresión regular simple para obtener solo palabras y los marcadores.
    """
    # r'<\/?:?s>|[A-Za-z]+' busca '<s>' o '</s>' o palabras (letras A-Z)
    return re.findall(r'<\/?:?s>|[A-Za-z]+', sentence)  # devuelve lista de tokens encontrados


def extract_bigrams(text):  # extrae todos los bigrams (pares consecutivos) del texto
    """
    1) Divide el texto en líneas (una oración por línea).
    2) Tokeniza cada línea.
    3) Crea pares consecutivos (bigrams) y los devuelve en una lista.
    """
    # Quitar espacios al inicio/final y separar por líneas
    sentences = text.strip().splitlines()  # lista de oraciones, una por línea

    bigrams = []  # lista donde guardaremos todos los pares

    for sentence in sentences:  # iterar cada oración
        tokens = tokenize_sentence(sentence)  # tokenizar la oración

        # Si la oración tiene menos de 2 tokens, no hay bigrams
        if len(tokens) < 2:
            continue  # saltar oraciones demasiado cortas

        # Recorremos los tokens y guardamos cada par consecutivo
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))  # añadir par (token_i, token_{i+1})

    return bigrams  # devolver todos los bigrams encontrados


if __name__ == "__main__":
    # Extraer bigrams del texto de ejemplo
    bigrams = extract_bigrams(text)

    # Contar la frecuencia de cada bigram (cuántas veces aparece)
    bigram_counts = Counter(bigrams)  # contador de pares (w1, w2)

    # Contar cuántas veces aparece cada palabra como contexto (palabra previa)
    context_counts = Counter([w1 for (w1, w2) in bigrams])  # cuenta de contextos w1

    # Calcular probabilidades condicionales P(w2 | w1) = count(w1,w2) / count(w1)
    bigram_probs = {}  # diccionario para guardar P(w2|w1)
    for (w1, w2), cnt in bigram_counts.items():  # para cada bigram y su frecuencia
        denom = context_counts[w1]  # número de veces que w1 aparece
        bigram_probs[(w1, w2)] = cnt / denom if denom > 0 else 0.0  # probabilidad condicional

    # Mostrar todos los bigrams en el orden en que aparecen
    print("Lista de bigrams (en orden):")
    for i, (w1, w2) in enumerate(bigrams, 1):
        # Número, palabra1, palabra2
        print(f"{i}. {w1} {w2}")  # imprime índice y el par de palabras

    # Mostrar conteo por bigram (ordenado para que sea fácil de leer)
    print("\nConteo de bigrams (bigram = frecuencia):")
    for bigram, count in sorted(bigram_counts.items()):
        print(f"('{bigram[0]}', '{bigram[1]}') = {count}")  # muestra cada bigram y su frecuencia

    # Mostrar conteo de contextos (cuántas veces aparece cada palabra como previa)
    print("\nConteo de contextos (palabra previa = frecuencia):")
    for ctx, cnt in sorted(context_counts.items()):
        print(f"'{ctx}' = {cnt}")  # muestra cada contexto y cuántas veces aparece

    # Mostrar probabilidades condicionales P(w2 | w1)
    print("\nProbabilidades condicionales P(w2 | w1):")
    # Ordenar por contexto y luego por palabra siguiente para lectura
    for (w1, w2), prob in sorted(bigram_probs.items()):
        print(f"P({w2} | {w1}) = {bigram_counts[(w1, w2)]}/{context_counts[w1]} = {prob:.2f}")  # imprime la fracción y la probabilidad

    # Resumen final: total y únicos
    print(f"\nTotal de bigrams: {len(bigrams)}")
    print(f"Bigram únicos: {len(bigram_counts)}")  # cuántos bigrams distintos hay

    # - Un "bigram" es un par de palabras consecutivas.
    # - Esto es una versión muy básica para entender el concepto.
    # - Si se quiere usar texto real, habría que limpiar signos de puntuación y mayúsculas.




