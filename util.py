import multiprocessing

import numpy as np
from tqdm import tqdm
import random
import main

random.seed(42)

def generate_pairs(encoded_sequences):
    num_sequences = len(encoded_sequences)
    pairs = []
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            encoded_sequence_x = encoded_sequences[i]
            encoded_sequence_y = encoded_sequences[j]
            pairs.append([encoded_sequence_x, encoded_sequence_y])
    return pairs

def encode_sequence(id, sequence, max_len):
    encoded = np.zeros((len(sequence), len(main.nucleotides)))
    for i, nucleotide in enumerate(sequence):
        encoded[i, main.nucleotides.index(nucleotide)] = 1
    padding = np.zeros((max_len - len(encoded), len(main.nucleotides)))
    encoded = np.vstack((encoded, padding))
    return id, encoded

import concurrent.futures

def add_noise(pairs, labels, data, num_workers=multiprocessing.cpu_count()):
    sequences_per_family = [[sequence for gene_id, sequence in value] for value in data.values()]
    threshold = len(pairs) * 2
    pairs_done = 0
    total_pairs = len(pairs)

    def process_pair(k):
        if k > threshold:
            return None

        i = random.randint(0, len(sequences_per_family) - 1)
        j = random.randint(0, len(sequences_per_family) - 1)
        if i != j:
            random_sequences1 = sequences_per_family[i]
            random_index1 = random.randint(0, len(random_sequences1) - 1)
            random_instance1 = random_sequences1[random_index1]
            random_sequences2 = sequences_per_family[j]
            random_index2 = random.randint(0, len(random_sequences2) - 1)
            random_instance2 = random_sequences2[random_index2]
            new_random_instance = np.asarray([random_instance1, random_instance2])
            labels.append(0)
            return new_random_instance, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_pair, range(pairs_done, total_pairs)), total=total_pairs - pairs_done,
                            desc='Adding Noise'))

    # Filter out None results (threshold exceeded)
    results = [result for result in results if result is not None]

    new_pairs, new_labels = zip(*results)

    # Convert labels to integers
    new_labels = [int(label) for label in new_labels]

    # Stack the pairs
    new_pairs = np.array(new_pairs)
    pairs = np.vstack((pairs, new_pairs))

    return pairs, np.array(labels)


